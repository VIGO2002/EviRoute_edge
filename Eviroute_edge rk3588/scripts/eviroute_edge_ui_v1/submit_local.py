#!/usr/bin/env python3
"""Create an offline web-UI task via loopback from SSH, no mouse necessary."""
import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def post(url, body, headers):
    req = Request(url, data=body, headers=headers, method='POST')
    with urlopen(req, timeout=90) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--arm', choices=('R_BCE', 'R_Debias'), default='R_BCE')
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--limit', type=int, default=12)
    args = p.parse_args()
    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(x for x in args.input.rglob('*') if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    if not files:
        p.error('No images in input')
    if args.limit < 1 or args.limit > 200:
        p.error('--limit must be 1..200')
    ids=[]
    for i, path in enumerate(files[:args.limit]):
        data=post(args.url+'/api/uploads',path.read_bytes(),{'Content-Type':'application/octet-stream','X-File-Name':quote(path.name, safe='')})
        ids.append(data['id'])
        print(f'Uploaded {i+1}/{min(len(files),args.limit)}: {path.name}')
    job=post(args.url+'/api/jobs',json.dumps({'arm':args.arm,'image_ids':ids}).encode(),{'Content-Type':'application/json'})
    print('JOB SUBMITTED:',job['id'],'model:',args.arm)
    print('Open Chromium at',args.url,'to view output; job runs on device')

if __name__=='__main__':
    main()
