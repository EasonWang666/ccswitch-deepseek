import http from 'http';

const postData = JSON.stringify({
    input: '请回复"测试成功"'
});

const req = http.request({
    hostname: '127.0.0.1',
    port: 11435,
    path: '/v1/responses',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
    }
}, (res) => {
    console.log('状态码:', res.statusCode);
    console.log('内容类型:', res.headers['content-type']);
    
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
        console.log('响应数据:', data);
    });
});

req.on('error', e => console.error('请求错误:', e.message));
req.write(postData);
req.end();
