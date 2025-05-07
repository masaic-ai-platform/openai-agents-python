
## Examples Built with OpenAI Agent SDK To Use Open Responses API Built In Tools

1. For SDK setup, see <a href="http://github.com/masaic-ai-platform/openai-agents-python?tab=readme-ov-file#get-started" target="_blank">Get Started</a>
2. For detailed instructions to run examples refer <a href="https://github.com/masaic-ai-platform/open-responses/blob/main/docs/Quickstart.md#6-running-agent-examples-built-with-openai-agent-sdk-to-use-open-responses-api-built-in-tools" target="_blank">Running Agent Examples</a>

## Running Open Telemetry Enabled Agent 
1. install required modules
```pip
pip install -r examples/open_responses/requirements-otel.txt
```

2. Set the following environment variables (if different from default values)  
- OTEL_EXPORTER_OTLP_ENDPOINT= http://localhost:4318
- OTEL_SERVICE_NAME= agent-4-OR
- OPEN_RESPONSES_URL=http://localhost:8080/v1
- GROQ_API_KEY=not_set

3. Run example.
```python
python -m examples.open_responses.agent_hands_off_with_otel
```
---