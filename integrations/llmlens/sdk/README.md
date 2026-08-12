# llmlens SDK

Instrument any LLM app to trace prompts, responses, latency, token cost, and
tool calls, and ship the spans to an llmlens server.

Intentionally tiny: the only hard dependency is `httpx`. Provider integrations
import their libraries lazily, so installing this pulls in nothing else.

```python
import llmlens

llmlens.configure(api_key="sk_...", url="http://localhost:8000")
llmlens.instrument("openai", "anthropic", "langchain")

with llmlens.trace("handle_request", user_id="u1"):
    ...   # nested spans and provider calls are captured automatically


@llmlens.observe()
def step():
    ...
```

`configure`, `instrument`, `trace`, `span`, `observe`, `set_user`,
`set_session`, `set_tags`, `callback_handler` and `flush` make up the public
API — see `src/llmlens/__init__.py`.

This file exists so the package is installable on its own. It used to point at
the parent repository's README via `readme = "../README.md"`, which hatchling
1.32 rejects: a readme path must resolve inside the project directory, whether
or not the file is there. That broke both `pip install -e sdk` in CI and the
API image build, which installs this package from a copied directory.

For the server, dashboard, and deployment docs, see the
[llmlens README](../README.md) one level up.
