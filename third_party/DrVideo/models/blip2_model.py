import torch
from PIL import Image
#from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class ImageCaptioner:
    def __init__(self,  model_name="blip2-opt", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.processor, self.model = self.initialize_model()
        
    def initialize_model(self):
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
            
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=False) 

        # for gpu with small memory
        elif self.model_name == "blip":
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
        else:
            raise NotImplementedError(f"{self.model_name} not implemented.")
        model.to(self.device)
        
        if self.device != 'cpu':
            model.half()
        return processor, model

    def image_caption(self, image, prompt, type_info):
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")
        # autoregressively complete prompt
        if type_info == "A" or type_info == "B":
            output = self.model.generate(**inputs, max_new_tokens=50)
        else:
            output = self.model.generate(**inputs, max_new_tokens=100)
        outputs = self.processor.decode(output[0], skip_special_tokens=True)
        outputs = outputs.strip()
        return outputs
    
    def image_caption_debug(self, image_src):
        return "A dish with salmon, broccoli, and something yellow."
