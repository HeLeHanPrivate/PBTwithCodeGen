import asyncio
import os
import httpx
import nest_asyncio  # import early to avoid duplicate calls
from openai import AsyncOpenAI
from typing import List, Dict, Any
import json
from tqdm import tqdm
from lcb_runner.runner.base_runner import BaseRunner


class LocalAPIRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)

        base_url = getattr(args, "api_base_url", None) or model.link
        is_loopback = "localhost" in base_url or "127.0.0.1" in base_url
        verify_ssl = not is_loopback and not getattr(args, "no_verify_ssl", False)
        
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(args.openai_timeout),
            verify=verify_ssl,
            trust_env=False,
            limits=httpx.Limits(max_connections=100)  # increase connection pool
        )

        api_key = self._resolve_api_key(args)
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self.http_client,
        )

        self.client_kwargs: Dict[str, Any] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            # "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,  # API limit
        }
        extra_body = self._build_extra_body(args)
        if extra_body:
            self.client_kwargs["extra_body"] = extra_body
        
        self.max_concurrency = getattr(args, 'max_concurrency', 8)

    @staticmethod
    def _resolve_api_key(args) -> str:
        if getattr(args, "api_key", None):
            return args.api_key
        env_name = getattr(args, "api_key_env", None)
        if env_name and os.getenv(env_name):
            return os.environ[env_name]
        if os.getenv("OPENAI_API_KEY"):
            return os.environ["OPENAI_API_KEY"]
        return "EMPTY"

    @staticmethod
    def _build_extra_body(args) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {}
        raw_kwargs = getattr(args, "chat_template_kwargs", None)
        chat_template_kwargs = None
        if raw_kwargs:
            try:
                chat_template_kwargs = json.loads(raw_kwargs)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--chat_template_kwargs must be valid JSON: {exc}") from exc
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError("--chat_template_kwargs must decode to a JSON object")
        if getattr(args, "disable_thinking", False):
            chat_template_kwargs = dict(chat_template_kwargs or {})
            chat_template_kwargs["enable_thinking"] = False
        elif chat_template_kwargs is None and "Qwen3.6" in getattr(args, "model", ""):
            chat_template_kwargs = {"enable_thinking": False}
        if chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        return extra_body

    @staticmethod
    def _prompt_cache_key(prompt):
        if isinstance(prompt, list):
            return json.dumps(prompt)
        if isinstance(prompt, tuple):
            return prompt[0] + json.dumps(prompt[1])
        return prompt

    @staticmethod
    def _normalize_prompt(prompt):
        if not isinstance(prompt, list):
            return [{'role': 'user', 'content': prompt}]
        return prompt

    def _run_single(self, prompt: List[Dict[str, str]]) -> List[str]:
        """Synchronous entry: run the coroutine via _run_async."""
        prompt = self._normalize_prompt(prompt)

        # choose coroutine based on n
        coro = (self._create_completion(prompt)
                if self.args.n == 1
                else self._run_parallel(prompt))

        # run async code through the helper method
        result = self._run_async(coro)
        return [result] if self.args.n == 1 else result

    async def _run_single_outputs_async(self, prompt: List[Dict[str, str]]) -> List[str]:
        prompt = self._normalize_prompt(prompt)
        if self.args.n == 1:
            return [await self._create_completion(prompt)]
        return await self._run_parallel(prompt)

    async def _run_batch_async(self, indexed_prompts):
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = {}

        async def run_indexed(index, prompt):
            async with semaphore:
                try:
                    return index, await self._run_single_outputs_async(prompt)
                except Exception as exc:
                    print(f"❌ Prompt {index} failed: {exc}")
                    return index, [""] * self.args.n

        tasks = [asyncio.create_task(run_indexed(index, prompt)) for index, prompt in indexed_prompts]
        with tqdm(total=len(tasks)) as progress:
            for task in asyncio.as_completed(tasks):
                index, output = await task
                results[index] = output
                progress.update(1)
        return results

    def run_batch(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
        outputs = [None for _ in prompts]
        uncached = []
        for index, prompt in enumerate(prompts):
            prompt_cache = self._prompt_cache_key(prompt)
            if (
                self.cache is not None
                and prompt_cache in self.cache
                and len(self.cache[prompt_cache]) == self.args.n
            ):
                outputs[index] = self.cache[prompt_cache]
            else:
                uncached.append((index, prompt))

        if uncached:
            batch_results = self._run_async(self._run_batch_async(uncached))
            for index, output in batch_results.items():
                assert len(output) == self.args.n
                outputs[index] = output

        if self.cache is not None:
            for prompt, output in zip(prompts, outputs):
                self.cache[self._prompt_cache_key(prompt)] = output

        return outputs

    def _run_async(self, coro):
        """Smart event-loop handler that works in scripts and Jupyter."""
        try:
            # check whether an event loop is already running
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # no running loop: plain script, use asyncio.run directly
            return asyncio.run(coro)
        else:
            # running loop exists (e.g. Jupyter): enable nested execution
            nest_asyncio.apply()
            return loop.run_until_complete(coro)

    async def _create_completion(self, prompt: List[Dict[str, str]]) -> str:
        """Single async request with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API call failed: {type(e).__name__}: {repr(e)}") from e
                print(f"⚠️  Retry {attempt+1}/{max_retries}: {type(e).__name__}: {repr(e)}")
                await asyncio.sleep(5 * (attempt + 1))

    async def _run_parallel(self, prompt: List[Dict[str, str]]) -> List[str]:
        """Run n requests in parallel (async mode)."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def fetch_with_semaphore():
            async with semaphore:
                return await self._create_completion(prompt)

        tasks = [fetch_with_semaphore() for _ in range(self.args.n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Change 1: use "" instead of " " so downstream logic does not treat it as valid content.
        # Change 2: final validation after gather with return_exceptions=True.
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Task {i+1}/{self.args.n} failed: {result}")
                processed_results.append("")  # placeholder for failed request
            else:
                processed_results.append(result)

        return processed_results

    def __del__(self):
        """Clean up the async HTTP client to avoid resource warnings."""
        try:
            # obtain the current loop (or create one) to close the client
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # schedule async close if the loop is already running
                loop.create_task(self.http_client.aclose())
            else:
                loop.run_until_complete(self.http_client.aclose())
        except:
            pass
