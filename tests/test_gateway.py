import unittest

import httpx

from gateway import app


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, handler, path='/lora/solve', payload=None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
            app.state.client = upstream
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                return await client.post(path, json=payload or {'question': '3*x - 7 = 20'})

    async def test_routes_and_forwards(self):
        for method, port in (('lora', 8001), ('qlora', 8000)):
            def handler(req):
                self.assertEqual(req.url.port, port)
                self.assertEqual(req.url.path, '/solve')
                self.assertIn(b'3*x - 7 = 20', req.content)
                return httpx.Response(200, json={'success': True, 'answer': 'x = 9'})
            response = await self.request(handler, f'/{method}/solve')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['method'], method)

    async def test_failures(self):
        for handler, status in (
            (lambda req: httpx.Response(503), 503),
            (lambda req: httpx.Response(200, text='invalid'), 502),
            (lambda req: httpx.Response(200, json={}), 502),
        ):
            self.assertEqual((await self.request(handler)).status_code, status)
        for error, status in ((httpx.ConnectError, 502), (httpx.ReadTimeout, 504)):
            def handler(req):
                raise error('failure', request=req)
            self.assertEqual((await self.request(handler)).status_code, status)

    async def test_invalid_requests_do_not_forward(self):
        def handler(req):
            self.fail('Must not call model')
        self.assertEqual((await self.request(handler, '/unknown/solve')).status_code, 404)
        self.assertEqual((await self.request(handler, payload={'question': ' '})).status_code, 422)

    async def test_cors(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.options('/lora/solve', headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type,ngrok-skip-browser-warning',
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:5173')


if __name__ == '__main__':
    unittest.main()
