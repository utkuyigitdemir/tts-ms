import asyncio
import httpx
import time

async def fetch(client, i):
    start = time.time()
    try:
        resp = await client.post(
            "http://127.0.0.1:8001/v1/audio/speech",
            json={"input": f"Merhaba, ben test kullanıcısı {i}.", "voice": "default"}
        )
        print(f"Req {i}: {resp.status_code} in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"Req {i} failed: {e}")

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Sending warmup...")
        await fetch(client, 0)
        
        print("Sending 5 concurrent requests...")
        start = time.time()
        tasks = [fetch(client, i) for i in range(1, 6)]
        await asyncio.gather(*tasks)
        print(f"Total batch time: {time.time()-start:.2f}s")
        
        try:
            resp = await client.get("http://127.0.0.1:8001/metrics")
            print("Metrics:")
            print(resp.json())
        except Exception as e:
            print("Could not fetch metrics:", e)

if __name__ == "__main__":
    asyncio.run(main())
