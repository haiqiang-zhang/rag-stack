import copy
import json
import logging
import random
from typing import Any, Dict, List

from rag_stack.ai_clients.base import AIClient
from rag_stack.dataset_generator.synthetic_article.domain_adapters.base import AbstractDomainAdapter

logger = logging.getLogger("RAG-Stack")

# Name pools for randomization (from RAGEval)
_CITY_NAMES = [
	'Springfield', 'Riverton', 'Clearwater', 'Sunnyvale', 'Greenfield', 'Fairview',
	'Hillcrest', 'Brookside', 'Lakewood', 'Cedarwood', 'Ridgewood', 'Roseville',
	'Maplewood', 'Pinehurst', 'Oakwood', 'Brighton', 'Kingsport', 'Lancaster',
	'Georgetown', 'Huntington', 'Madison', 'Norwood', 'Summerville', 'Windsor',
	'Ashton', 'Belmont', 'Cameron', 'Danbury', 'Eastwood', 'Foxboro', 'Granville',
	'Hartford', 'Inglewood', 'Jacksonville', 'Kensington', 'Lexington', 'Milton',
	'Newport', 'Orchard', 'Preston', 'Quincy', 'Richmond', 'Sterling', 'Tiverton',
	'Unionville', 'Vermont', 'Westwood', 'Yorktown',
]
_SURNAMES = [
	'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
	'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
	'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
	'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
	'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King',
	'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green',
	'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
	'Carter', 'Roberts',
]
_INITIALS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
_MONTHS = [
	'January', 'February', 'March', 'April', 'May', 'June',
	'July', 'August', 'September', 'October', 'November', 'December',
]
# Chinese name pools
_ZH_SURNAMES = [
	'王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴',
	'徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
	'梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧',
	'程', '曹', '袁', '邓', '许', '傅', '沈', '曾', '彭', '吕',
	'苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎',
]
_ZH_MONTHS = [
	'1月', '2月', '3月', '4月', '5月', '6月',
	'7月', '8月', '9月', '10月', '11月', '12月',
]
_ZH_CITIES = [
	'碧波市', '星河市', '云海市', '翠竹市', '金沙市', '银川市',
	'玉泉市', '清风市', '明月市', '青山市', '绿水市', '红枫市',
]
_ZH_STREETS = [
	'铁桥街', '长安路', '朝阳路', '建设路', '人民路', '解放路',
	'中山路', '和平路', '光明路', '文化路', '科技路', '学院路',
]

# Diseases that may cause death
_MAY_DIED_EN = [
	'Bacterial Food Poisoning', 'Hepatitis A', 'Malaria',
	'Nasopharyngeal Malignancy', 'Esophageal Malignancy',
	'Epidemic Cerebrospinal Meningitis', 'Pneumonia',
	'Congenital Heart Disease', 'Neonatal Tetanus',
]
_MAY_DIED_ZH = [
	'细菌性食物中毒', '甲型肝炎', '疟疾',
	'鼻咽恶性肿瘤', '食管恶性肿瘤',
	'流行性脑脊髓膜炎', '肺炎',
	'先天性心脏病', '新生儿破伤风',
]


class MedicalAdapter(AbstractDomainAdapter):
	domain = "medical"

	def load_schema_templates(self) -> Dict[str, Any]:
		"""Load surgery and non-surgery hospitalization templates."""
		return {
			"surgery": self._load_json("zhuyuan_shoushu.json"),
			"no_surgery": self._load_json("zhuyuan_bushoushu.json"),
		}

	def load_topic_catalog(self) -> Dict[str, List[dict]]:
		"""Load disease type → disease items mapping."""
		return self._load_json("disease_type_name_detail.json")

	async def generate_config(
		self,
		ai_client: AIClient,
		topic_type: str,
		topic_item: dict,
		schema_template: dict,
		random_seed: int = 42,
	) -> dict:
		"""Generate a filled medical record config (s1 logic).

		1. Pick the right template (surgery vs no-surgery)
		2. Randomize demographics
		3. Call LLM to fill remaining fields
		"""
		rng = random.Random(random_seed)

		# Always start from surgery template then conditionally remove fields
		schema = copy.deepcopy(schema_template["surgery"])

		# Determine surgery/death status
		may_died = _MAY_DIED_EN if self.lang == "en" else _MAY_DIED_ZH
		disease_name = topic_item.get("Name", topic_item.get("名称", ""))

		is_dead = rng.random() > 0.8 if disease_name in may_died else False
		is_surgery = True if is_dead else rng.random() > 0.8

		# Fill demographics
		schema = self._fill_demographics(schema, topic_type, topic_item, is_surgery, is_dead, rng)

		# Build prompt from JSONL template
		article_prompts = self.load_article_prompts()
		s1_prompt = article_prompts["s1_config"]
		if self.lang == "en":
			surgery_text = "had surgery" if is_surgery else "did not have surgery"
			death_text = "is deceased" if is_dead else "is not deceased"
		else:
			surgery_text = "做了手术" if is_surgery else "未做手术"
			death_text = "已死亡" if is_dead else "未死亡"
		user_prompt = s1_prompt["user_prompt"].format(
			disease_name=disease_name,
			disease_detail=json.dumps(topic_item, ensure_ascii=False),
			surgery_text=surgery_text,
			death_text=death_text,
			config=json.dumps(schema, ensure_ascii=False, indent=1),
		)
		messages = [
			{"role": "system", "content": s1_prompt["system_prompt"]},
			{"role": "user", "content": user_prompt},
		]

		max_retries = 5
		for attempt in range(max_retries):
			response_text = await ai_client.chat(messages)
			try:
				config = self._parse_json_response(response_text)
				config["DiseaseType"] = {topic_type: topic_item}
				return config
			except (json.JSONDecodeError, ValueError) as e:
				logger.warning(f"Config JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}")
				if attempt == max_retries - 1:
					raise

	async def generate_article(
		self, ai_client: AIClient, config: dict, topic_type: str, topic_item: dict
	) -> str:
		"""Convert config JSON to narrative medical record text (s2 logic)."""
		disease_name = topic_item.get("Name", topic_item.get("名称", ""))
		article_prompts = self.load_article_prompts()
		s2_prompt = article_prompts["s2_article"]
		user_prompt = s2_prompt["user_prompt"].format(
			disease_name=f"{topic_type}-{disease_name}",
			disease_detail=json.dumps(topic_item, ensure_ascii=False),
			config=json.dumps(config, ensure_ascii=False, indent=1),
		)
		system_prompt = s2_prompt["system_prompt"]
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]
		return await ai_client.chat(messages)

	def _fill_demographics(
		self, schema: dict, topic_type: str, topic_item: dict,
		is_surgery: bool, is_dead: bool, rng: random.Random,
	) -> dict:
		"""Fill in random patient demographics (ported from RAGEval set_value)."""
		record_key = "Hospitalization Record" if self.lang == "en" else "住院病历"
		basic_key = "Basic Information" if self.lang == "en" else "基本信息"
		post_key = "Post-Admission Course Records" if self.lang == "en" else "入院后病程记录"

		record = schema.get(record_key, schema)
		basic = record.get(basic_key, {})

		if self.lang == "en":
			basic["Name"] = rng.choice(_INITIALS) + '. ' + rng.choice(_SURNAMES)
			basic["Gender"] = rng.choice(["male", "female"])
			if topic_type == "Pregnancy, Childbirth, and the Puerperium Complications":
				basic["Gender"] = "female"
			if basic["Gender"] == "male":
				marital_key = "Marital and Family History"
				if marital_key in record and "Menstrual History" in record[marital_key]:
					del record[marital_key]["Menstrual History"]
			basic["Age"] = str(rng.randint(1, 100))
			if topic_type == "Conditions Originating in the Perinatal Period":
				basic["Age"] = f"{rng.randint(1, 10)} hours after birth"
			day = rng.randint(1, 28)
			suffix = {1: "st", 2: "nd", 3: "rd"}.get(day, "th")
			basic["Admission Time"] = f"{day}{suffix}, {rng.choice(_MONTHS)}"
			city = rng.choice(_CITY_NAMES)
			street = rng.choice([n for n in _CITY_NAMES if n != city])
			basic["Address"] = f"{rng.randint(1, 100)}, {street} street, {city}"
		else:
			basic["姓名"] = rng.choice(_ZH_SURNAMES) + "某某"
			basic["性别"] = rng.choice(["男", "女"])
			if "妊娠" in topic_type or "产褥" in topic_type:
				basic["性别"] = "女"
			if basic["性别"] == "男":
				marital_key = "婚育史、月经史"
				if marital_key in record and "月经史" in record[marital_key]:
					del record[marital_key]["月经史"]
			basic["年龄"] = f"{rng.randint(1, 100)}岁"
			if "围生期" in topic_type:
				basic["年龄"] = f"出生后{rng.randint(1, 10)}小时"
			basic["入院时间"] = f"{rng.choice(_ZH_MONTHS)}{rng.randint(1, 28)}日"
			city = rng.choice(_ZH_CITIES)
			street = rng.choice(_ZH_STREETS)
			basic["地址"] = f"{city}{street}{rng.randint(1, 100)}号"

		# Remove death/surgery sections based on status
		if post_key in record:
			death_key = "Death Related" if self.lang == "en" else "死亡相关"
			surgery_key = "Surgery Related" if self.lang == "en" else "手术相关"
			if not is_dead and death_key in record[post_key]:
				del record[post_key][death_key]
			if not is_surgery and surgery_key in record[post_key]:
				del record[post_key][surgery_key]

		return schema

	@staticmethod
	def _parse_json_response(response_text: str) -> dict:
		"""Extract JSON from LLM response, handling markdown wrappers and extra data."""
		text = response_text.strip()
		# Remove markdown code block if present
		if "```json" in text:
			text = text.split("```json", 1)[1]
			if "```" in text:
				text = text.split("```", 1)[0]
			text = text.strip()
		elif text.startswith("```"):
			text = text.split("\n", 1)[1] if "\n" in text else text[3:]
			if "```" in text:
				text = text.split("```", 1)[0]
			text = text.strip()
		# Find the first complete JSON object using bracket matching
		start = text.find("{")
		if start == -1:
			return json.loads(text)
		depth = 0
		in_string = False
		escape = False
		for i in range(start, len(text)):
			c = text[i]
			if escape:
				escape = False
				continue
			if c == '\\':
				escape = True
				continue
			if c == '"' and not escape:
				in_string = not in_string
				continue
			if in_string:
				continue
			if c == '{':
				depth += 1
			elif c == '}':
				depth -= 1
				if depth == 0:
					return json.loads(text[start:i + 1])
		# Fallback
		return json.loads(text[start:])
