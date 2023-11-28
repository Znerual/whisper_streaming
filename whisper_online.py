#!/usr/bin/env python3
import sys
import numpy as np
import librosa  
from functools import lru_cache
import time
import Levenshtein


@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


# Whisper backend

class ASRBase:

    # join transcribe words with this character (" " for whisper_timestamped, "" for faster-whisper because it emits the spaces when neeeded)
    sep = " "

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, **kwargs):
        self.transcribe_kargs = {}
        self.original_language = lan 

        self.model = self.load_model(modelsize, cache_dir, model_dir, **kwargs)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


## requires imports:
#      import whisper
#      import whisper_timestamped

class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.

    If used, requires imports:
        import whisper
        import whisper_timestamped
    """

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        if model_dir is not None:
            print("ignoring model_dir, not implemented",file=sys.stderr)
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = whisper_timestamped.transcribe_timestamped(self.model, audio, language=self.original_language, initial_prompt=init_prompt, verbose=None, condition_on_previous_text=True)
        return result
 
    def ts_words(self,r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"],w["end"],w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        raise NotImplemented("Feature use_vad is not implemented for whisper_timestamped backend.")


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.

    Requires imports, if used:
        import faster_whisper
    """

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None, gpu=True):
        from faster_whisper import WhisperModel


        if model_dir is not None:
            print(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.",file=sys.stderr)
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")


        # this worked fast and reliably on NVIDIA L40
        if gpu:
            model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)
        else:
            model = WhisperModel(model_size_or_path, device="cpu", compute_type="float32", download_root=cache_dir)
        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        #model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
#        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt=""):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        return segments

    def ts_words(self, segment):
        #for segment in segments:
        for word in segment.words:
            # not stripping the spaces -- should not be merged with them!
            w = word.word
            t = (word.start, word.end, w, word.probability)
            yield t


    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"



class HypothesisBuffer:

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

    def insert_single(self, new, offset):
        a,b,t,p = new
        if len(self.commited_in_buffer) == 0 or a + offset > self.commited_in_buffer[-1][1]-0.1:
            self.commited_in_buffer.append((a+offset,b+offset,t,p))
            self.last_commited_time = b+offset
            self.last_commited_word = t
            return (a+offset,b+offset,t,p)
        
        if a + offset < self.commited_in_buffer[-1][1] and a + offset > self.commited_in_buffer[-1][0]-0.1:
            if p > self.commited_in_buffer[-1][3]:
                print("Replaced last word in buffer:",self.commited_in_buffer[-1],"with",new,file=sys.stderr)
                self.commited_in_buffer[-1] = (a+offset,b+offset,t,p)
                self.last_commited_time = b+offset
                self.last_commited_word = t
                return (a+offset,b+offset,t,p)
            
        return None

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        
        new = [(a+offset,b+offset,t,p) for a,b,t,p in new]
        self.new = [(a,b,t,p) for a,b,t,p in new if a > self.last_commited_time-0.1]

        #print("new:",new)
        #print("self.new:",self.new)

        if len(self.new) >= 1:
            a,b,t,p = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if Levenshtein.distance(c.lower(), tail.lower()) < 2:
                            print("removing last",i,"words:",file=sys.stderr)
                            for j in range(i):
                                print("\t",self.new.pop(0),file=sys.stderr)
                            break
                        
        #self.buffer.extend(self.new)
        self.commited_in_buffer.extend(self.new)
        self.last_commited_time = self.new[-1][1]
        return self.new
            
        
    def complete(self):
        return self.buffer

class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer, language):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.language = language
        self.init()

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer()
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer. 
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0,len(self.commited)-1)

        if k == 0:
            return "", ""
        
        #print(self.commited,file=sys.stderr)
        #print(self.commited[:k], file=sys.stderr)
        p = self.commited[:k]
        p = [t for _,_,t,_ in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t,_ in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (commited) partial transcript.
        """

        
        while self.buffer_time_offset < len(self.audio_buffer) // self.SAMPLING_RATE - 0.2:
            prompt, non_prompt = self.prompt()
            #print("PROMPT:", prompt, file=sys.stderr)
            #print("CONTEXT:", non_prompt, file=sys.stderr)
            
            index_start = int(self.buffer_time_offset * self.SAMPLING_RATE)
            index_end = len(self.audio_buffer)
            if len(self.audio_buffer[index_start:index_end])/self.SAMPLING_RATE > 30:
                index_end = index_start + int(30*self.SAMPLING_RATE)
            
            #print(f"transcribing {len(self.audio_buffer[index_start:index_end])/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset + self.last_chunked_at:2.2f}",file=sys.stderr)
            print(f"Index start: {index_start}, index end: {index_end}",file=sys.stderr)
            segments = self.asr.transcribe(self.audio_buffer[index_start:index_end], init_prompt=prompt)
            #print("res: ", res)
            # transform to [(beg,end,"word1"), ...]
            for segment in segments:
                for t in self.asr.ts_words(segment):
                    #print(t)
                    o = self.transcript_buffer.insert_single(t, self.buffer_time_offset)
                    if not o is None:
                        self.commited.append(o)
                        yield o

      
            
            #print(">>>>COMPLETE NOW:",self.to_flush(o),file=sys.stderr,flush=True)
            #print("INCOMPLETE:",self.to_flush(self.transcript_buffer.complete()),file=sys.stderr,flush=True)

            self.buffer_time_offset += min(int((index_end - index_start) / self.SAMPLING_RATE), 28) # 2 seconds overlap
            print("buffer time offset:",self.buffer_time_offset,file=sys.stderr,flush=True)
            # there is a newly confirmed text
            
            #yield self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []: 
            print("Nothing comminted")
            return
        print(self.commited,file=sys.stderr)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            print("\t\tSENT:",s,file=sys.stderr)
        if len(sents) < 2:
            print("Not enough sentences")
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        #print(f"--- sentence chunked at {chunk_at:2.2f}",file=sys.stderr)
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []: 
            #print("Nothing comminted")
            return

        ends = self.asr.segments_end_ts(res)

        t = self.buffer_time_offset + self.last_chunked_at #self.commited[-1][1]
        
        while len(ends) > 1:

            
            e = ends[-1]+self.last_chunked_at
                
            if e <= t:
                #print(f"--- segment chunked at {e:2.2f}",file=sys.stderr)
                self.chunk_at(e)
                return
            
            # too far, let's try the previous segment
            ends.pop(-1)
            # else:
            #     print(f"--- last segment not within commited area",file=sys.stderr)
                
        
        #print(f"--- not enough segments to chunk",file=sys.stderr)





    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        #self.transcript_buffer.pop_commited(time)
        print(f"Chunking at time: {time}, with a last chunked at {self.last_chunked_at} and a buffer time offset of {self.buffer_time_offset}",file=sys.stderr)
        cut_seconds = time - self.last_chunked_at
        self.audio_buffer = self.audio_buffer[int(cut_seconds)*self.SAMPLING_RATE:]
        self.buffer_time_offset -= cut_seconds
        self.last_chunked_at = time
        # print("cut at ",time,file=sys.stderr)
        # print("Removed ",cut_seconds," seconds from audio buffer",file=sys.stderr)
        # print("Chnked, buffer time offset: ",self.buffer_time_offset,file=sys.stderr)

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        if hasattr(self.tokenizer, "split"):
            s = self.tokenizer.split(t)
        elif hasattr(self.tokenizer, "tokenize"):
            s = self.tokenizer.tokenize(t)
        else:
            raise ValueError("tokenizer must have split or tokenize method")
        
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w,p = cwords.pop(0)
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        print("last, noncommited:",f,file=sys.stderr)
        return f


    def to_flush(self, sents, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)

WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from sacremoses import MosesTokenizer
        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        print(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.", file=sys.stderr)
        lan = None

    from wtpsplit import WtP
    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()

