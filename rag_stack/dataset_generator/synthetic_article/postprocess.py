"""Postprocess functions copied from RAGEval with minimal changes.

Changes from RAGEval original:
- gpt_client.generate([{...}])[0] → await ai_client.chat(msgs)
- Merged en/zh into single functions with lang parameter
- Removed dotenv/OpenAIClient imports
- Added MAX_RETRIES safety cap (RAGEval uses while True which can hang with weaker models)
"""
import json
import logging
from typing import List

from rag_stack.ai_clients.base import AIClient

logger = logging.getLogger("RAG-Stack")

MAX_RETRIES = 2


async def postprocess_en(response, system_prompt: str, user_prompt: str, ai_client: AIClient) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    output = []
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    if 'ref' in item:
                        try:
                            if "Unable to answer" in item['answer'] or item['ref'] == []:
                                item['question type'] = 'Irrelevant Unsolvable Question'
                                item['ref'] = []
                            json_obj = {
                                "question type": item['question type'],
                                "question": item['question'],
                                "answer": item['answer'],
                                "ref": item['ref'],
                            }
                            output.append(json_obj)
                        except (KeyError, ValueError, IndexError):
                            flag = False
                            break
                    else:
                        try:
                            if "Unable to answer" in item['answer']:
                                item['question type'] = 'Irrelevant Unsolvable Question'
                            json_obj = {
                                "question type": item['question type'],
                                "question": item['question'],
                                "answer": item['answer'],
                            }
                            output.append(json_obj)
                        except (KeyError, ValueError, IndexError):
                            flag = False
                            break
            except json.JSONDecodeError:
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return output
    logger.error(f"postprocess_en failed after {MAX_RETRIES} retries")
    return []


async def postprocess_zh(response, system_prompt: str, user_prompt: str, ai_client: AIClient) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    output = []
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    if 'ref' in item:
                        try:
                            if "无法回答" in item['答案'] or item['ref'] == []:
                                item['问题类型'] = '无关无解问'
                                item['ref'] = []
                            json_obj = {
                                "问题类型": item['问题类型'],
                                "问题": item['问题'],
                                "答案": item['答案'],
                                "ref": item['ref'],
                            }
                            output.append(json_obj)
                        except (KeyError, ValueError, IndexError):
                            flag = False
                            break
                    else:
                        try:
                            if "无法回答" in item['答案']:
                                item['问题类型'] = '无关无解问'
                            json_obj = {
                                "问题类型": item['问题类型'],
                                "问题": item['问题'],
                                "答案": item['答案'],
                            }
                            output.append(json_obj)
                        except (KeyError, ValueError, IndexError):
                            flag = False
                            break
            except json.JSONDecodeError:
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return output
    logger.error(f"postprocess_zh failed after {MAX_RETRIES} retries")
    return []


async def postprocess_reference_check_single_doc_en(response: str, new_sentences: List, system_prompt: str, user_prompt: str, ai_client: AIClient) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    refs = []
                    try:
                        if item['ref'] == [] and item['question type'] != 'Irrelevant Unsolvable Question':
                            flag = False
                        else:
                            for ref in item['ref']:
                                refs.append(new_sentences[int(ref) - 1].split('] ')[-1])
                            item['ref'] = refs
                    except (KeyError, ValueError, IndexError):
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return response
    logger.error(f"postprocess_reference_check_single_doc_en failed after {MAX_RETRIES} retries")
    return []


async def postprocess_reference_check_single_doc_zh(response: str, new_sentences: List, system_prompt: str, user_prompt: str, ai_client: AIClient) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    refs = []
                    try:
                        if item['ref'] == [] and item['问题类型'] != '无关无解问':
                            flag = False
                        else:
                            for ref in item['ref']:
                                refs.append(new_sentences[int(ref) - 1].split('] ')[-1])
                            item['ref'] = refs
                    except (KeyError, ValueError, IndexError):
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return response
    logger.error(f"postprocess_reference_check_single_doc_zh failed after {MAX_RETRIES} retries")
    return []


async def postprocess_reference_check_multi_doc_en(response: str, domain: str, new_sentences_1: List, new_sentences_2: List, system_prompt: str, user_prompt: str, ai_client: AIClient) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        for ref in item['ref']:
                            if ref['content'] == [] and item['question type'] != 'Irrelevant Unanswerable Question':
                                flag = False
                                break
                            else:
                                if domain == 'finance':
                                    for sentence in new_sentences_1:
                                        if ref['company_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['company_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break

                                if domain == 'law':
                                    for sentence in new_sentences_1:
                                        if ref['court_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['court_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break

                                if domain == 'medical':
                                    for sentence in new_sentences_1:
                                        if ref['hospital_patient_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['hospital_patient_name'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                    except (KeyError, ValueError, IndexError):
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return response
    logger.error(f"postprocess_reference_check_multi_doc_en failed after {MAX_RETRIES} retries")
    return []


async def postprocess_reference_check_multi_doc_zh(response: str, domain: str, new_sentences_1: List, new_sentences_2: List, system_prompt: str, user_prompt: str, ai_client: AIClient) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format, transform number to sentences to avoid errors when saving as a JSON file and maybe sometimes gpt-4o will generate ref without any number.
    """
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        for ref in item['ref']:
                            if ref['content'] == [] and item['问题类型'] != '无关无解问':
                                flag = False
                                break
                            else:
                                if domain == 'finance':
                                    for sentence in new_sentences_1:
                                        if ref['公司名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['公司名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break

                                if domain == 'law':
                                    for sentence in new_sentences_1:
                                        if ref['法院名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['法院名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break

                                if domain == 'medical':
                                    for sentence in new_sentences_1:
                                        if ref['医院_病人名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_1[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                                    for sentence in new_sentences_2:
                                        if ref['医院_病人名'] in sentence:
                                            refs = []
                                            for r in ref['content']:
                                                refs.append(new_sentences_2[int(r) - 1].split('] ')[-1])
                                            ref['content'] = refs
                                            break
                    except (KeyError, ValueError, IndexError):
                        flag = False
                        break
            except json.JSONDecodeError:
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return response
    logger.error(f"postprocess_reference_check_multi_doc_zh failed after {MAX_RETRIES} retries")
    return []


async def postprocess_irrelevant_en(response: str, system_prompt: str, user_prompt: str, ai_client: AIClient, domain: str, name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    output = []
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        if "Unable to answer" in item['answer'] or item['ref'] == []:
                            item['question type'] = 'Irrelevant Unsolvable Question'
                            item['ref'] = []
                        if domain == 'law':
                            json_obj = {
                                "court_name": item['court_name'],
                                "question type": item['question type'],
                                "question": item['question'],
                                "answer": item['answer'],
                                "ref": item['ref'],
                            }
                            output.append(json_obj)
                        if domain == 'medical':
                            if item['hospital_patient_name'] != name:
                                logger.warning(f"[postprocess_irrelevant_en] retry {_retry+1}: hospital_patient_name mismatch: got '{item.get('hospital_patient_name')}', expected '{name}'")
                                flag = False
                                break
                            else:
                                json_obj = {
                                    "hospital_patient_name": item['hospital_patient_name'],
                                    "question type": item['question type'],
                                    "question": item['question'],
                                    "answer": item['answer'],
                                    "ref": item['ref'],
                                }
                                output.append(json_obj)
                    except (KeyError, ValueError, IndexError) as e:
                        logger.warning(f"[postprocess_irrelevant_en] retry {_retry+1}: {type(e).__name__}: {e} | item keys: {list(item.keys()) if isinstance(item, dict) else 'not a dict'}")
                        flag = False
                        break
            except json.JSONDecodeError as e:
                logger.warning(f"[postprocess_irrelevant_en] retry {_retry+1}: JSONDecodeError: {e} | response preview: {repr(response[:200]) if isinstance(response, str) else type(response)}")
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return output
    logger.error(f"postprocess_irrelevant_en failed after {MAX_RETRIES} retries")
    return []


async def postprocess_irrelevant_zh(response: str, system_prompt: str, user_prompt: str, ai_client: AIClient, domain: str, name: str) -> List:
    """
    Remove common extra characters in gpt-4o, check the question type and array format to avoid errors when saving as a JSON file.
    """
    output = []
    flag = True
    for _retry in range(MAX_RETRIES):
        if "```json\n" in response:
            response = response.replace("```json\n", "").replace("```", "")
        else:
            try:
                response = json.loads(response)
                for item in response:
                    try:
                        if "无法回答" in item['答案'] or item['ref'] == []:
                            item['问题类型'] = '无关无解问'
                            item['ref'] = []
                        if domain == 'law':
                            if item['法院名'] != name:
                                logger.warning(f"[postprocess_irrelevant_zh] retry {_retry+1}: 法院名 mismatch: got '{item.get('法院名')}', expected '{name}'")
                                flag = False
                                break
                            else:
                                json_obj = {
                                    "法院名": item['法院名'],
                                    "问题类型": item['问题类型'],
                                    "问题": item['问题'],
                                    "答案": item['答案'],
                                    "ref": item['ref'],
                                }
                                output.append(json_obj)
                        if domain == 'medical':
                            if item['医院_病人名'] != name:
                                logger.warning(f"[postprocess_irrelevant_zh] retry {_retry+1}: 医院_病人名 mismatch: got '{item.get('医院_病人名')}', expected '{name}'")
                                flag = False
                                break
                            else:
                                json_obj = {
                                    "医院_病人名": item['医院_病人名'],
                                    "问题类型": item['问题类型'],
                                    "问题": item['问题'],
                                    "答案": item['答案'],
                                    "ref": item['ref'],
                                }
                                output.append(json_obj)
                    except (KeyError, ValueError, IndexError) as e:
                        logger.warning(f"[postprocess_irrelevant_zh] retry {_retry+1}: {type(e).__name__}: {e} | item keys: {list(item.keys()) if isinstance(item, dict) else 'not a dict'}")
                        flag = False
                        break
            except json.JSONDecodeError as e:
                logger.warning(f"[postprocess_irrelevant_zh] retry {_retry+1}: JSONDecodeError: {e} | response preview: {repr(response[:200]) if isinstance(response, str) else type(response)}")
                flag = False

            if not flag:
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = await ai_client.chat(msgs)
                flag = True
            else:
                return output
    logger.error(f"postprocess_irrelevant_zh failed after {MAX_RETRIES} retries")
    return []


# Convenience wrapper used by qa_generator.py
async def postprocess_qa(response: str, system_prompt: str, user_prompt: str, ai_client: AIClient, lang: str = "en") -> List:
    if lang == "en":
        return await postprocess_en(response, system_prompt, user_prompt, ai_client)
    else:
        return await postprocess_zh(response, system_prompt, user_prompt, ai_client)
