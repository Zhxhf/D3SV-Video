from string import Template


def first_char_as_answer(res):
    mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    if res[0] in mapping:
        return mapping[res[0]]
    return -1

def identity(res):
    return res

def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        anchor_index = res.find(anchor)
        pred = -1  # if decoding failed, return -1
        if anchor_index >= 0:
            pred_letter = res[anchor_index+len(anchor)]
            if pred_letter in mapping:
                pred = mapping[pred_letter]
        return pred
    return f

def get_intervals_as_list(text):
    text = text.split('.')[0]
    text = text.strip()
    if text[-1] != ']':
        index = text.rfind(']')
        assert index > 0
        text = text[:index+1]
    interval_list_text = text.split('and')
    intervals = []
    for interval_text in interval_list_text:
        if ',' not in interval_text:
            intervals.append([0, 0])
            continue
        start_text, end_text = interval_text.split(',')
        start_text, end_text = start_text.strip(' []'), end_text.strip(' []')
        if start_text == 'None':
            start_text = '0'
        if end_text == 'None':
            end_text = '1'
        start, end = int(start_text), int(end_text)
        intervals.append([start, end])
    return intervals


class PromptTemplate(object):
    def __init__(self, head, template, post_process_fn):
        self.head = head
        self.prompt_template = template
        self.post_process_fn = post_process_fn

    def get_num_stages(self):
        return len(self.template)

    def get_template_str(self):
        template = []
        for temp in self.prompt_template:
            template.append(temp.safe_substitute())
        return template

    def fill(self, **kwargs):
        # match variable names: duration, narration, question, optionA, optionB, optionC, optionD, optionE, num_words
        prompt_filled = []
        for temp in self.prompt_template:
            prompt_filled.append(temp.substitute(kwargs))
        return prompt_filled

class PromptFactory(object):
    def __init__(self):
        self.prompt_templates = self.build()
    
    def build(self):
        prompt_templates = {}

        prompt_templates['rag_llm'] = PromptTemplate(
            head = "You are a helpful expert in select correct question related video segment from a video",
            template = [
                Template("""Our goal is to identify video segment that contain crucial information necessary for answering the question. These video segment
                        should address the query. Given a video that has some video segments. To answer the following question:
                        '''
                        ${question_text}
                        '''
                        We provide some candidate video segments which are shown in below:
                        '''
                        ${context}
                        '''
                        The prefix '#C' refer to the camera wearer, while prefix '#O' refer to someone other than the camera wearer. you need to select the correct video segments that contain crucial information necessary for answering the question through reasoning from these video segments. To help you judge better, we also provide task related information prompts from another intelligent agent as follows:
                        '''
                        ${gpt_prompt}
                        '''
                        Please think step-by-step and select the correct video segment and drop the incorrect video segment, and your answer must be the index of video segment(such as 1, 2, 3, 4...) rather than other content. You must not provide any other response or explanation.
                        """
                )
            ],
            post_process_fn = identity
        )

        prompt_templates['llm_reasoning'] = PromptTemplate(
            head = "You are a helpful expert to answer a multiple-choice question related to a video",
            template = [
                Template("""You are tasked with answering a multiple-choice question related to a video. Given the following descriptions of the question related video clips in the video:\n
                        '''
                        ${context}
                        '''
                        Please answer the following question:\n
                        '''
                        ${question}
                        '''
                        Here are the choices.\n A: ${option1}\n B: ${option2}\n C: ${option3}\n D: ${option4}\n E: ${option5}\n
                        The question has 5 choices, labeled as A, B, C, D, E. Please think step-by-step and write the best answer index in Json format {'final_answer': 'xxx'}, if the information is too vague to provide an accurate answer, make
                        your best guess. \n
                        Note your final answer must be one of the letters (A, B, C, D, or E) and the output must be the following format. You must not provide any other response or explanation.
                        {'final_answer': 'xxx'}
                        """
                )
            ],
            post_process_fn = identity
        )

        prompt_templates['rag_stage1'] = PromptTemplate(
            head = "You are a helpful expert in video understanding",
            template = [
                Template("""You are given some language descriptions of a first-person view video and one question about the video. Each video is 3 minute long. Each sentence describes a frame. Here are the descriptions: \n
                         '''
                         ${captions}
                         '''
                         Here is the question: \n
                         '''
                         ${question_context}
                         '''
                         Please first pinpoint the timestamps that best respond to the questions provided. The timestamp consist of multiple frames. Ensure each answer encompasses not just the event in question but also the relevant context before and after. In your responses to questions about past events, follow these guidelines: \n
                         (a). Incorporate Context: Expand your answers to include not just the central event but also the context preceding and following it. \n
                         (b). Unify Related Actions: When a question requires a sequence of actions, such as 'Where did I put the scarf after I closed the door?', merge all relevant events into a single interval that conveys the full story. \n
                         (c). Opt for Broad Understanding: Favor comprehensive intervals that cover all relevant details over more precise but less informative ones.\n
                         After providing a time interval, please give me a 50 words summary for each response. When doing summarization, remember that your summary will be used to answer this question: \n
                         '''
                         ${question_summary_round1}
                         '''
                         Finally, please also give me a 20 words summary for each timestamps not selected. For example, the video has 180 frames and ranges from 1-180, if the timestamps that best respond to the questions are in 23-39, 57-69, you should first summarize the context with 50 words from frame 23 to frame 39 and summarize the context with 50 words from frame 57 to frame 69 which used to answer the question. Then you should summarize the context with 20 words from frame 1 to frame 22, summarize the context with 20 words from frame 40 to frame 56, and summarize the context with 20 words from frame 70 to frame 180. Finally the output will be as follows:\n
                         [{'duration': 'frame 1 - frame 22', 'summary': 'xxx'}, {'duration': 'frame 23 - frame 39', 'summary': 'xxx'}, {'duration': 'frame 40 - frame 56', 'summary': 'xxx'}, {'duration': 'frame 57 - frame 69', 'summary': 'xxx'}, {'duration': 'frame 70 - frame 180', 'summary': 'xxx'}]\n
                         Please note the consistency of duration, such as 1-22, 23-39, 40-56, 57-69, 70-180. Please note the timestamps 23-39, 57-69 is only a example and has nothing to do with your task. The timestamps consist of multiple frames. Please remember that your 50 words summary will be used to answer this question when doing summarization in question related key timestamps and give a 20 words summary of the timestamps not selected.\n
                         Please follow the output format as below:\n
                         ['duration': 'frame xxx - frame xxx', 'summary': 'xxx'}]\n

                         You must not provide any other response or explanation
                         """
                )
            ],
            post_process_fn = identity
        )

        prompt_templates['loop'] = PromptTemplate(
            head = "You are a helpful expert in video understanding",
            template = [
                Template("""You are given some language descriptions of a first-person view video along with a question about the video. Your task is to determine whether these descriptions can answer the multi-choice question accurately, comprehensively, reasonably, and without contradiction.\n
                         If the descriptions do not provide enough information to answer the multi-choice question, it indicates that detailed descriptions are missing for some frames of the video. You will need to specify which frames require more detailed information. Here are the details you need to understand before giving your output:\n
                         1. The video is 3 minutes long, containing a total of 180 frames.\n
                         2. Each sentence in these language descriptions represents the text description for a single frame.\n
                         3. The format of each sentence is {frame id, general caption, dense caption}. The frame id indicates the temporal position of the frame, ranging from 1 to 180. The general caption provides a simple and short description of the frame. The dense caption provides a detailed description of the frame. Some frames have not been provided with a dense caption; these are the frames you may consider as potentially missing information. Frames with dense captions are considered to have no missing information.\n
                         4. The images which have been provided dense captions are these (${potential_key_frames}).\n
                         Now, here are the language descriptions of the video:\n
                         '''  
                         ${captions}
                         '''
                         Here is the multi-choice question: \n
                         '''
                         ${question_context}
                         '''
                         Here are the choices.\n A: ${option1}\n B: ${option2}\n C: ${option3}\n D: ${option4}\n E: ${option5}\n
                         The answer of the multi-choice question is one of the above choices.\n
                         Here is the output template:\n
                         If you conclude that the descriptions are sufficient to answer the above multi-choice question and give a reasonable explanation for the corresponding choice about the video, your output should be as follows: {'confidence': '1', 'add_frames': 'N/A'}\n
                         If you find the descriptions insufficient, indicating a confidence level of 0, you should identify the frames that need more detailed information and these frames can help answer the multi-choice question accurately, comprehensively, reasonably. The output should be as follows: {'confidence': '0', 'add_frames': [xx,xx,xx,xx]}\n
                         Please note that frame selections range from 1 to 180, and frames that already have a dense caption cannot be selected. The output format must be \n
                         {'confidence': '0'/'1', 'add_frames': [xxx,xxx]/'N/A'}\n
                         You must not provide any other response or explanation. Now, generate your output:
                         """
                )
            ],
            post_process_fn = identity
        )

        prompt_templates['judge'] = PromptTemplate(
            head = "You are a helpful expert in video understanding",
            template = [
                Template("""You are given some language descriptions of a first-person view video along with a question about the video. \n
                            1.The video is 3 minutes long, containing a total of 180 frames.\n
                            2. Each sentence in these language descriptions represents the text description for a single frame.\n
                            3.  The format of each sentence is {frame id, description}. The frame id indicates the temporal position of the frame, ranging from 1 to 180.\n
                            Here are the original descriptions of this video:\n
                            '''  
                            ${captions}
                            '''
                            Here is the question:\n
                            ''' 
                            ${question_context}
                            '''
                            \n
                            ${gpt_prompt}
                            \n
                            Your task is to determine whether these descriptions above can answer the question accurately. \n
                            If your answer is yes, please give me an reasonable explanation. the output will be as follows:\n
                            {'confidence': '1', 'explanation': ["xxxx"]}\n
                            If your answer is no, the confidence is 0, indicating the provided description is insufficient. Please give me a reasonable explanation for what frame is missing. For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements(e.g., objects, humans, interactions, actions, and scenes) in the explanation. The output will be as follows:\n
                            {'confidence': '0', 'explanation': ["xxxx"]}\n
                            You must not provide any other response or explanation.
                         """
                )
            ],
            post_process_fn = identity
        )


        prompt_templates['find'] = PromptTemplate(
            head = "You are a helpful expert in video understanding",
            template = [
                Template("""You are given some language descriptions of a first-person view video along with a question about the video. \n
                            1.The video is 3 minutes long, containing a total of 180 frames.\n
                            2. Each sentence in these language descriptions represents the text description for a single frame.\n
                            3.  The format of each sentence is {frame id, description}. The frame id indicates the temporal position of the frame, ranging from 1 to 180.\n
                            Here are the original descriptions of this video:\n
                            '''  
                            ${captions}
                            '''
                            To answer the following question:\n
                            ''' 
                            ${question_context}
                            '''
                            However, theses descriptions are insufficient and cannot answer this question accurately, reasonably, and without contradiction.\n
                            ${explanation}
                            Your task is to determine which frame needs which type of information and can answer this question accurately, reasonably, and without contradiction. \n
                            The two types of information are as follows:\n
                            A: Given an image, get a detailed description of the image (image caption, just like what is shown in this image?)\n
                            B: Given an image, get a response to the above question (visual question answering)\n

                            Please note that frame selections range from 1 to 180 and no more 3 frames. These frames (${type_A}) already have type A information and these frames (${type_B}) already have type B information, please note not to repeatedly select this type of information from these frames. The output must be as follows:
                            [{'frame': '1/2/3…/180', 'type': 'A/B'}]\n
                            You must not provide any other response or explanation.
                         """
                )
            ],
            post_process_fn = identity
        )

        prompt_templates['qa_standard'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n"),
            ],
            post_process_fn = first_char_as_answer
        )

        # egoschema QA (raw captions as input) few-shot
        prompt_templates['qa_standard_fewshot'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is $duration seconds long. You are also given a question and five potential choices. Your task is to answer with a correct choice based on the video descriptions. \nHere are a few examples. \n${examplars}\n\n Now answer this question.\nDescriptions: ${narration}.\n Question: ${question}\n A: ${optionA}.\n B: ${optionB}.\n C: ${optionC}.\n D: ${optionD}.\n E: ${optionE}.\n Answer: "),
            ],
            post_process_fn = first_char_as_answer
        )
        
        # egoschema QA (summary as input)
        prompt_templates['qa_sum'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n"),
            ],
            post_process_fn = first_char_as_answer
        )

        # egoschema sum (standard)
        prompt_templates['sum_standard'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template('You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n Please give me a $num_words words summary.')
            ],
            post_process_fn = identity
        )

        # egoschema sum (q)
        prompt_templates['sum_q'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template('You are given some language descriptions of a first person view video. The video is ${duration_new} seconds long. Each sentence describes a 1s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. Here are the descriptions: $narration_new.\n Please give me a 300 words summary. When doing summarization, remember that your summary will be used to answer this multiple choice question: $question_new'),
            ],
            post_process_fn = identity
        )

        prompt_templates['sum_dense'] = PromptTemplate(
            head = "You are a helpful expert in video analysis.",
            template = [
                Template('You are given some language descriptions of a video. The video is 3 minutes long, containing a total of 180 frames. Each sentence in these language descriptions represents the text description for a single frame. The format of each sentence is {frame id, caption}.\n Here are the descriptions: $narration_new.\n Please give me a 200 words summary. When doing summarization, remember that your summary will be used to answer this multiple choice question: $question_new'),
            ],
            post_process_fn = identity
        )


        # egoschema sum (qa)
        prompt_templates['sum_qa'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template('You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n Please give me a $num_words words summary. When doing summarization, remember that your summary will be used to answer this multiple choice question: $question\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n Do not answer this question directly. Instead, use the question and choices to guide your summary.')
            ],
            post_process_fn = identity
        )

        # egoschema QA zero-shot-CoT
        prompt_templates['qa_zs-cot'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n Before answering this question, let's think step by step."),
                Template("Please provide a single-letter answer (A, B, C, D, E) to the multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. Your response should only contain one letter.")
            ],
            post_process_fn = first_char_as_answer
        )

        # egoschema QA plan-and-solve
        prompt_templates['qa_plansolve'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n To answer this question, let's first prepare relevant information and decompose it into 3 sub-questions. Then, let's answer the sub-questions one by one. Finally, let's answer the multiple choice question."),
                Template("Please provide a single-letter answer (A, B, C, D, E) to the multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. Your response should only contain one letter.")
            ],
            post_process_fn = first_char_as_answer
        )

        # next-qa QA, intentQA QA
        prompt_templates['qa_next'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. If you are not sure, answer with the most likely answer. You are given some language descriptions of a first person view video. The video is 1 FPS and the descriptions are the captions every 2 frames. Each caption starts with the frame number.\nHere are the descriptions:\n$narration\n Here is the question: $question?\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n"),
            ],
            post_process_fn = first_char_as_answer
        )

        # next-gqa GQA
        prompt_templates['gqa'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("I will provide video descriptions and one question about the video. The video is 1 FPS and the descriptions are the captions every 2 frames. Each caption starts with the frame number.\n To answer this question, what is the minimun frame interval to check?\n Follow this format: [frame_start_index, frame_end_index]. Do not provide any explanation.\n Here are the descriptions:\n$narration\n Here is the question: $question?\n Please follow the output format as follows:\n #Example1: [5, 19]\n #Example2: [30, 60]\n #Example3: [1, 10] and [50, 60]"),
            ],
            post_process_fn = get_intervals_as_list
        )

        # egoschema QA llama
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        anchor = 'The most correct answer is ('
        prompt_templates['qa_standard_llama'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly." + E_SYS + 'Here are the descriptions:\n$narration\n Here is the question: $question.\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n' + E_INST + anchor),
            ],
            post_process_fn = first_char_after_anchor(anchor)
        )

        return prompt_templates

    def get(self, prompt_type):
        return self.prompt_templates[prompt_type]
    
















