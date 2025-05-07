# client_tracing.py

import os
import asyncio

from openai import AsyncOpenAI
from agents import Agent, Runner, set_tracing_disabled
from agents.models.openai_responses import OpenAIResponsesModel

# ── OpenTelemetry setup ────────────────────────────────────────────────────────
from opentelemetry import trace, propagate
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# 1) Read OTLP endpoint & service name from env
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
service_name = os.getenv("OTEL_SERVICE_NAME", "agent-4-OR")

# 2) Configure TracerProvider + OTLP exporter
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: service_name
})
provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# 3) Ensure W3C (traceparent) is the global propagator
propagate.set_global_textmap(propagate.get_global_textmap())


# ── HTTPX event hook for inject+log ────────────────────────────────────────────
from httpx import Request, AsyncClient

async def inject_and_log(request: Request):
    # Inject W3C headers into the outgoing HTTPX request
    propagate.inject(request.headers)
    # Print them so you can see exactly what was sent
    print(f"→ OUT {request.method} {request.url}\n   headers: {dict(request.headers)}")


# ── Build your AsyncOpenAI client & wire in the hook ──────────────────────────
BASE_URL = os.getenv("OPEN_RESPONSES_URL") or "http://localhost:8080/v1" #Either set OPEN_RESPONSES_URL in environment variable or put it directly here.
API_KEY = os.getenv("GROQ_API_KEY") or "" #Either set GROQ_API_KEY in environment variable or put it directly here.
MODEL_NAME = "groq@qwen-qwq-32b"

# Create the AsyncOpenAI client
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    default_headers={"Authorization": f"Bearer {API_KEY}"}
)

# Grab its underlying HTTPX client and register our hook
httpx_client: AsyncClient = client._client
httpx_client.event_hooks.setdefault("request", []).append(inject_and_log)


# ── Build your agents (as before) ─────────────────────────────────────────────
set_tracing_disabled(disabled=False)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client)
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English.",
    model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client)
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client)
)


# ── Run the workflow inside a root span ───────────────────────────────────────
async def main():
    with tracer.start_as_current_span("triage-workflow"):
        result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
        print("Final Output:", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())