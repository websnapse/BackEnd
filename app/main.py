import asyncio
from fastapi import FastAPI, WebSocket
from app.errors import WebsnapseError
from app.models import SNPSystem, Regular
from app.utils import validate_rule
from fastapi.middleware.cors import CORSMiddleware
from app.SNP import MatrixSNPSystem

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://websnapse.github.io",
]
app.add_middleware(  # type: ignore
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

status = {
    1: "spiking",
    2: "forgetting",
    0: "default",
    -1: "closed",
}

resume_event = asyncio.Event()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/simulate")
async def simulate_all(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.pseudorandom_simulate_all()

    print(matrixSNP.states)
    print(matrixSNP.contents)

    return {
        "states": matrixSNP.states.tolist(),
        "configurations": matrixSNP.contents.tolist(),
        "keys": matrixSNP.neuron_keys,
    }


@app.post("/simulate/last")
async def simulate_all(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.pseudorandom_simulate_all()

    print(matrixSNP.content)

    return {
        "contents": matrixSNP.content.tolist(),
    }


@app.post("/simulate/step")
async def simulate_step(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.compute_next_configuration()

    print(matrixSNP.state)
    print(matrixSNP.content)
    print(matrixSNP.halted)
    return {
        "states": matrixSNP.state.tolist(),
        "configurations": matrixSNP.content.tolist(),
        "keys": matrixSNP.neuron_keys,
        "halted": bool(matrixSNP.halted),
    }


async def prev(websocket: WebSocket, matrixSNP: MatrixSNPSystem):
    matrixSNP.compute_prev_configuration()

    configs = []

    for index, key in enumerate(matrixSNP.neuron_keys):
        configs.append(
            {
                "id": key,
                "content": matrixSNP.content[index],
                "delay": int(matrixSNP.delay[index]),
                "state": status[matrixSNP.state[index]],
            }
        )

    try:
        await websocket.send_json(
            {
                "type": "step",
                "configurations": configs,
                "halted": bool(matrixSNP.halted),
                "tick": int(matrixSNP.cursor),
            }
        )
    except Exception as e:
        print(e)


async def next_guided(websocket: WebSocket, matrixSNP: MatrixSNPSystem, speed: int):
    matrixSNP.compute_spikeable_mx()
    choices = matrixSNP.spikeable_mx.shape[0]
    if choices > 1:
        spikeable = matrixSNP.check_non_determinism()
        await websocket.send_json({"type": "prompt", "choices": spikeable})
        await resume_event.wait()
        resume_event.clear()
    else:
        matrixSNP.decision_vct = matrixSNP.spikeable_mx[0]
    matrixSNP.compute_next_configuration()

    configs = []

    for index, key in enumerate(matrixSNP.neuron_keys):
        configs.append(
            {
                "id": key,
                "content": matrixSNP.content[index],
                "delay": int(matrixSNP.delay[index]),
                "state": status[matrixSNP.state[index]],
            }
        )

    try:
        await websocket.send_json(
            {
                "type": "step",
                "configurations": configs,
                "halted": bool(matrixSNP.halted),
                "tick": int(matrixSNP.cursor),
            }
        )
    except Exception as e:
        print(e)
    finally:
        if matrixSNP.halted:
            return


@app.post("/validate")
async def validate_neuron(neuron: Regular):
    try:

        for index, rule in enumerate(neuron.rules):
            try:
                validate_rule(rule)
            except WebsnapseError as e:
                return {
                    "type": "error",
                    "message": f"Invalid rule ${rule}$ found at position {index + 1}",
                }
        return {"type": "success", "message": "Neuron validated successfully"}
    except WebsnapseError as e:
        return {"type": "error", "message": e.message}
    except Exception as e:
        print(e)
        return {"type": "error", "message": str(e)}


@app.websocket("/ws/simulate/guided")
async def guided_mode(websocket: WebSocket):
    await websocket.accept()

    try:
        req = await websocket.receive_json()
        system = req["data"]
        speed = req["speed"]
        matrixSNP = MatrixSNPSystem(SNPSystem(**system))
        simulating = True
        simulating_task = asyncio.create_task(next_guided(websocket, matrixSNP, speed))
        while True:
            try:
                data = await websocket.receive_json()
                cmd = data["cmd"]

                if cmd == "stop" and simulating:
                    simulating_task.cancel()
                    simulating = False
                elif cmd == "continue" and not simulating:
                    simulating_task = asyncio.create_task(
                        next_guided(websocket, matrixSNP, speed)
                    )
                elif cmd == "next":
                    simulating_task.cancel()
                    simulating_task = asyncio.create_task(
                        next_guided(websocket, matrixSNP, speed)
                    )
                elif cmd == "choice":
                    choice = data["choice"]
                    matrixSNP.create_spiking_vector(choice)
                    resume_event.set()
                elif cmd == "history":
                    await websocket.send_json(
                        {
                            "type": "history",
                            "history": matrixSNP.decisions.tolist(),
                        }
                    )
                elif cmd == "prev":
                    simulating_task.cancel()
                    simulating_task = asyncio.create_task(prev(websocket, matrixSNP))

                elif cmd == "speed":
                    speed = data["speed"]
                    simulating_task.cancel()
                    if simulating:
                        simulating_task = asyncio.create_task(
                            next_guided(websocket, matrixSNP, speed)
                        )
                elif cmd == "received":
                    simulating_task.cancel()
                    await asyncio.sleep(1 / speed)
                    simulating_task = asyncio.create_task(
                        next_guided(websocket, matrixSNP, speed)
                    )

            except KeyError:
                await websocket.send_json(
                    {"type": "error", "message": "Command not recognized"}
                )
            except:
                break
    except Exception as e:
        pass


async def next_pseudorandom(
    websocket: WebSocket, matrixSNP: MatrixSNPSystem, speed: int
):
    matrixSNP.pseudorandom_simulate_next()

    configs = []

    for index, key in enumerate(matrixSNP.neuron_keys):
        configs.append(
            {
                "id": key,
                "content": matrixSNP.content[index],
                "delay": int(matrixSNP.delay[index]),
                "state": status[matrixSNP.state[index]],
            }
        )

    try:
        await websocket.send_json(
            {
                "type": "step",
                "configurations": configs,
                "halted": bool(matrixSNP.halted),
                "tick": int(matrixSNP.cursor),
            }
        )
    except Exception as e:
        print(e)
    finally:
        if matrixSNP.halted:
            return


@app.websocket("/ws/simulate/pseudorandom")
async def pseudorandom_mode(websocket: WebSocket):
    await websocket.accept()

    try:
        req = await websocket.receive_json()
        system = req["data"]
        speed = req["speed"]
        matrixSNP = MatrixSNPSystem(SNPSystem(**system))
        simulating = True
        simulating_task = asyncio.create_task(
            next_pseudorandom(websocket, matrixSNP, speed)
        )
        while True:
            try:
                data = await websocket.receive_json()
                cmd = data["cmd"]

                if cmd == "stop" and simulating:
                    simulating_task.cancel()
                    simulating = False
                elif cmd == "continue" and not simulating:
                    simulating_task = asyncio.create_task(
                        next_pseudorandom(websocket, matrixSNP, speed)
                    )
                elif cmd == "next":
                    simulating_task.cancel()
                    simulating_task = asyncio.create_task(
                        next_pseudorandom(websocket, matrixSNP, speed)
                    )
                elif cmd == "history":
                    await websocket.send_json(
                        {
                            "type": "history",
                            "history": matrixSNP.decisions.tolist(),
                        }
                    )
                elif cmd == "prev":
                    simulating_task.cancel()
                    simulating_task = asyncio.create_task(prev(websocket, matrixSNP))
                elif cmd == "speed":
                    speed = data["speed"]
                    simulating_task.cancel()
                    if simulating:
                        simulating_task = asyncio.create_task(
                            next_pseudorandom(websocket, matrixSNP, speed)
                        )
                elif cmd == "received":
                    simulating_task.cancel()
                    await asyncio.sleep(1 / speed)
                    simulating_task = asyncio.create_task(
                        next_pseudorandom(websocket, matrixSNP, speed)
                    )
            except KeyError:
                await websocket.send_json(
                    {"type": "error", "message": "Command not recognized"}
                )
            except WebsnapseError as e:
                await websocket.send_json({"type": "error", "message": e.message})
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
