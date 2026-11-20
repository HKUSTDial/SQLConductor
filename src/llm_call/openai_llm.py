from openai import OpenAI
import dotenv
from typing import List, Optional
from src.llm_call.cost_recoder import CostRecorder
import time

dotenv.load_dotenv(override=True)

DEFAULT_COST_RECORDER = CostRecorder(model="gpt-3.5-turbo")

MAX_RETRYING_TIMES = 5


N_CALLING_STRATEGY_SINGLE = "single"
N_CALLING_STRATEGY_MULTIPLE = "multiple"

def call_openai(prompt: str,
                model: str,
                temperature: float = 0.0,
                top_p: float = 1.0,
                n: int = 1,
                max_tokens: int = 512,
                stop: List[str] = None,
                base_url: str = None,
                api_key: str = None,
                n_strategy: str = N_CALLING_STRATEGY_SINGLE,
                cost_recorder: Optional[CostRecorder] = DEFAULT_COST_RECORDER) -> str:
    client = OpenAI(
        api_key=api_key if api_key is not None else "dummy",
        base_url=base_url if base_url is not None else None,
    )
    retrying = 0
    while retrying < MAX_RETRYING_TIMES:
        try:
            if n == 1 or (n > 1 and n_strategy == N_CALLING_STRATEGY_SINGLE):
                _t0 = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                    stop=stop,
                )
                print(f"[LLM] model={model} n={n} elapsed={time.time()-_t0:.2f}s prompt_tokens={response.usage.prompt_tokens} completion_tokens={response.usage.completion_tokens}")
                if cost_recorder is not None:
                    cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                contents = [choice.message.content for choice in response.choices]
                break
            elif n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                contents = []
                for _i in range(n):
                    _t0 = time.time()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        top_p=top_p,
                        stop=stop,
                    )
                    print(f"[LLM] model={model} call={_i+1}/{n} elapsed={time.time()-_t0:.2f}s prompt_tokens={response.usage.prompt_tokens} completion_tokens={response.usage.completion_tokens}")
                    if cost_recorder is not None:
                        cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    contents.append(response.choices[0].message.content)
                break
            else:
                raise ValueError(f"Invalid n_strategy: {n_strategy} for n: {n}")
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            print(f"Retry attempt {retrying + 1}")
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                raise e
            time.sleep(10)
    return contents
