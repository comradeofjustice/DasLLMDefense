import re
from typing import Optional, List, Dict, Tuple, Union

import autogen
import openai
from autogen import Agent, UserProxyAgent, OpenAIWrapper

from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense, DefenseAssistantAgent, \
    DefenseGroupChat
from defense.utility import load_defense_prompt
from evaluator.evaluate_helper import evaluate_defense_with_response


class CoordinatorAgent(DefenseAssistantAgent):
    def __init__(self, **kwargs):
        kwargs["name"] = "Coordinator"
        self.defense_strategy_name = kwargs.pop("defense_strategy_name")
        super().__init__(**kwargs)
        self.defense_prompt = load_defense_prompt()
        self.register_reply([Agent, None], CoordinatorAgent.generate_coordinate_reply)

    def generate_coordinate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        last_agent_name = self._oai_messages[sender][-1]['name']
        if last_agent_name == "TaskInputAgent":
            final = True
            response = self.defense_prompt[self.defense_strategy_name]["1_user"]
        elif last_agent_name == "IntentionAnalyzer":
            final = True
            # 安全地提取system_input，避免IndexError
            find_results = re.findall(r"--SYSTEM INPUT START--\n((.|\n)*)--SYSTEM INPUT END--",
                                      self._oai_messages[sender][0]['content'])
            if find_results:
                system_input = find_results[0][0]
            else:
                # 如果找不到标记，使用整个内容或提供默认值
                system_input = self._oai_messages[sender][0]['content']
            response = self.defense_prompt[self.defense_strategy_name]["2_user"].replace("[INSERT INPUT HERE]", system_input)
        elif last_agent_name == "Judge":
            final = True
            response = "Next: TaskInputAgent"
        else:
            raise ValueError("Unknown agent name", last_agent_name)
        return final, response


class AutoGenDetectorV1:
    def __init__(self, config_list, defense_strategy_name="explicit_2_agent"):
        self.defense_prompt = load_defense_prompt()
        config = {
            "cache_seed": 123,  # change the cache_seed for different trials
            "config_list": config_list,
            "timeout": 300,
        }
        self.user_input_agent = UserProxyAgent(name="TaskInputAgent",
                                               human_input_mode="NEVER",
                                               is_termination_msg=lambda x: True)
        self.coordinator = CoordinatorAgent(
            llm_config=config,
            defense_strategy_name=defense_strategy_name
        )
        self.analyzer = DefenseAssistantAgent(
            name="IntentionAnalyzer",
            system_message=self.defense_prompt[defense_strategy_name]["1_system"],
            llm_config=config
        )
        self.judge = DefenseAssistantAgent(
            name="Judge",
            system_message=self.defense_prompt[defense_strategy_name]["2_system"],
            llm_config=config
        )
        groupchat = DefenseGroupChat(
            agents=[self.user_input_agent, self.coordinator, self.analyzer, self.judge],
            messages=[], max_round=10)
        self.manager = autogen.GroupChatManager(groupchat=groupchat,
                                                llm_config=config)
        self.defense_strategy_name = defense_strategy_name

    def clear_history(self):
        self.user_input_agent.clear_history()
        self.analyzer.clear_history()
        self.judge.clear_history()
        self.coordinator.clear_history()
        self.manager.clear_history()

    def __call__(self, content_text, user_prompt=None):
        self.clear_history()
        try:
            self.user_input_agent.initiate_chat(self.manager,
                                                message=self.defense_prompt[self.defense_strategy_name]["0_user"].
                                                replace("[INSERT INPUT HERE]", content_text),
                                                clear_history=True, silent=True)
            llm_judgement_output = self.manager._oai_messages[self.judge][-2]['content']
            success = True
            
            # 首先检查LLM的判断
            if "INVALID" in llm_judgement_output:
                valid = False
            elif "VALID" in llm_judgement_output:
                valid = True
                # 增加额外的委婉拒绝检测逻辑
                # 仅在LLM判断为VALID时进行二次检验
                refusal_patterns = [
                    "i'm sorry",
                    "i cannot",
                    "i can't",
                    "i won't be able to",
                    "unable to",
                    "not able to",
                    "mission and goals",
                    "please feel free to contact me",
                    "apologize",
                    "您好，我很抱歉",
                    "我不会支持任何形式的暴力或不良行为",
                    "无法回答",
                    "不能回答",
                    "很抱歉",
                    "如果你有任何疑问或需要帮助"
                ]
                
                # 检查原始内容文本中是否包含委婉拒绝的表达
                content_lower = content_text.lower()
                print(f"Checking content_text: {content_text}")
                print(f"Checking content_lower: {content_lower}")
                for pattern in refusal_patterns:
                    if pattern in content_lower or pattern in content_text:
                        valid = False
                        print(f"检测到委婉拒绝模式: {pattern}")
                        llm_judgement_output="INVALID:[DASGuard]:你好，此问题涉嫌违规，暂时不能帮您回答这个问题,判断依据为"+llm_judgement_output
                        break
            else:
                valid = False
                success = False
                    
        except openai.BadRequestError as e:
            valid = False
            success = True
            llm_judgement_output = "ERROR" + str(e)

        return valid, success, llm_judgement_output

    def agency_execute(self, llm_output):
        valid, success, llm_output = self(llm_output)
        return llm_output


class AutoGenDetectorV0125(AutoGenDetectorV1):
    def __init__(self, config_list):
        super().__init__(config_list, defense_strategy_name="explicit_2_agent_v0125")


if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument("--log_file", type=str, default="results/detection_summary.json")
    # args.add_argument("--port", type=int, default=8005)
    # args = args.parse_args()
    #
    # detector = AutoGenDetectorV1(port=args.port)
    # evaluate_explicit_detector(detector, log_file=args.log_file)

    evaluate_defense_with_response(task_agency=AutoGenDetectorV1,
                                   defense_agency=ExplicitMultiAgentDefense,
                                   defense_output_name="data/defense_output/openai/gpt-3.5-turbo-1106/ex-2.json",
                                   model_name="gpt-3.5-turbo-1106")
