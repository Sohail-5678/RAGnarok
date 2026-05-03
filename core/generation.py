"""
Multi-LLM generation layer with dynamic routing.
Supports OpenAI, Google Gemini, Groq, and Anthropic Claude.
"""
import os
import asyncio
from typing import Optional, Generator, AsyncGenerator

import streamlit as st

from config import LLM_PROVIDERS, SYSTEM_PROMPT, NO_CONTEXT_PROMPT, LLMConfig


# ─── LLM Client Factory ──────────────────────────────────────────────

def get_llm_config(
    provider: str,
    api_keys: dict,
) -> Optional[LLMConfig]:
    """Build LLM config based on provider and available API keys."""
    provider_info = LLM_PROVIDERS.get(provider)
    if not provider_info:
        return None

    # Check for API key: user-provided first, then environment variable
    api_key = api_keys.get(provider, "").strip()
    if not api_key:
        api_key = os.environ.get(provider_info["env_key"], "").strip()

    if not api_key:
        return None

    return LLMConfig(
        provider=provider,
        model_name=provider_info["model"],
        api_key=api_key,
    )


def get_available_providers(api_keys: dict) -> list:
    """Return list of providers that have valid API keys."""
    available = []
    for provider_id, info in LLM_PROVIDERS.items():
        key = api_keys.get(provider_id, "").strip()
        if not key:
            key = os.environ.get(info["env_key"], "").strip()
        if key:
            available.append(provider_id)
    return available


# ─── Generation Functions ─────────────────────────────────────────────

def generate_response_openai(
    question: str,
    context: str,
    config: LLMConfig,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """Generate streaming response using OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key)

    if context:
        system_msg = SYSTEM_PROMPT.format(context=context, question=question)
    else:
        system_msg = NO_CONTEXT_PROMPT.format(question=question)

    messages = [{"role": "system", "content": system_msg}]

    # Add chat history
    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_response_groq(
    question: str,
    context: str,
    config: LLMConfig,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """Generate streaming response using Groq."""
    from groq import Groq

    client = Groq(api_key=config.api_key)

    if context:
        system_msg = SYSTEM_PROMPT.format(context=context, question=question)
    else:
        system_msg = NO_CONTEXT_PROMPT.format(question=question)

    messages = [{"role": "system", "content": system_msg}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_response_gemini(
    question: str,
    context: str,
    config: LLMConfig,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """Generate streaming response using Google Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(
        config.model_name,
        generation_config=genai.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        ),
    )

    if context:
        prompt = SYSTEM_PROMPT.format(context=context, question=question)
    else:
        prompt = NO_CONTEXT_PROMPT.format(question=question)

    # Add history context
    if chat_history:
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in chat_history[-6:]
        )
        prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"

    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        if chunk.text:
            yield chunk.text


def generate_response_claude(
    question: str,
    context: str,
    config: LLMConfig,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """Generate streaming response using Anthropic Claude."""
    from anthropic import Anthropic

    client = Anthropic(api_key=config.api_key)

    if context:
        system_msg = SYSTEM_PROMPT.format(context=context, question=question)
    else:
        system_msg = NO_CONTEXT_PROMPT.format(question=question)

    messages = []
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    with client.messages.stream(
        model=config.model_name,
        system=system_msg,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─── Unified Generation Router ────────────────────────────────────────

def generate_response(
    question: str,
    context: str,
    provider: str,
    api_keys: dict,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """
    Route to the appropriate LLM provider and generate a streaming response.
    
    Falls back to Groq if the selected provider is unavailable.
    """
    config = get_llm_config(provider, api_keys)

    # Fallback chain: selected → groq → any available
    if not config:
        config = get_llm_config("groq", api_keys)
    if not config:
        available = get_available_providers(api_keys)
        if available:
            config = get_llm_config(available[0], api_keys)

    if not config:
        yield "⚠️ No API key configured. Please add at least one API key in the sidebar settings."
        return

    # Route to the correct generator
    generators = {
        "openai": generate_response_openai,
        "groq": generate_response_groq,
        "gemini": generate_response_gemini,
        "claude": generate_response_claude,
    }

    generator_fn = generators.get(config.provider)
    if not generator_fn:
        yield f"⚠️ Unsupported provider: {config.provider}"
        return

    try:
        yield from generator_fn(question, context, config, chat_history)
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            yield f"🔑 Authentication error with {config.provider}. Please check your API key."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            yield f"⏳ Rate limit reached for {config.provider}. Please wait a moment and try again."
        else:
            yield f"❌ Error from {config.provider}: {error_msg[:300]}"
