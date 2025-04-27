# lambda/index.py
import json
import os
import urllib.request
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError
import urllib

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# 環境変数からFastAPIのURLを取得 or デフォルト値を設定
FASTAPI_URL = os.environ.get('FASTAPI_URL','https://7ae4-34-87-37-54.ngrok-free.app')

def lambda_handler(event, context):
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)

        # FastAPIからモデル名を取得
        req = urllib.request.Request(FASTAPI_URL + '/health', method='GET', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as res:
            if res.status != 200:
                raise Exception("Failed to get model name from FastAPI")
            response_body = json.loads(res.read())
            print("FastAPI response /health:", response_body)
            # print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # モデル用のリクエストペイロードを構築
        # 会話履歴を含める
        # ロール付きで履歴をテキスト化
        history_prompt = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        ) + "\nassistant:"
        
        # FastAPI用のリクエストペイロード
        request_body = {
            "prompt": history_prompt,
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }

        # FastAPIにリクエストを送信
        req = urllib.request.Request(FASTAPI_URL + '/generate', json.dumps(request_body).encode(), headers={'Content-Type': 'application/json', 'accept': 'application/json'})
        print("Calling Bedrock invoke_model API with payload:", json.dumps(request_body).encode())
        with urllib.request.urlopen(req) as res:
            if res.status != 200:
                raise Exception("Failed to post generate from FastAPI")
            response_body = json.loads(res.read())
            print("FastAPI response /generate:", response_body)
        
        # アシスタントの応答を取得
        assistant_response = response_body['generated_text']
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
