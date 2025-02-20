import os
import gradio as gr
import json
from openai import OpenAI
import base64
import tempfile
import shutil
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import requests
from PIL import Image
import io
import time
from pathlib import Path

# テンポラリディレクトリの作成
TEMP_DIR = './temp'
os.makedirs(TEMP_DIR, exist_ok=True)

# OpenAI クライアントの初期化（APIキーは後で設定）
client = OpenAI(
    api_key="OPENAI_API_KEY" ,  
    base_url="OPENAI_ENDPOINTS" 
)

# 環境変数をセット
os.environ['DATABRICKS'] = 'DATABRICKS_ENVIRON'
os.environ['DATABRICKS'] = 'DATABRICKS_ENVIRON'

def encode_image(image_path):
    """画像をBase64エンコードする関数"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

import time

def find_similar_coordinations(coordination_text, max_retries=5, wait_time=10):
    """ベクトル検索で類似のコーディネーションを見つける関数（リトライ対応）"""
    for attempt in range(max_retries):
        try:
            vsc = VectorSearchClient(disable_notice=True)
            vs_index = vsc.get_index(
                endpoint_name="vs_endpoint",
                index_name="dev.haruna_osaki.fashion_documentation_vs_index"
            )

            results = vs_index.similarity_search(
                query_text=coordination_text,
                columns=["ID", "Detail", "Category", "Color", "Path"],
                num_results=1,
                filters={}
            )

            returned_docs = []
            docs = results.get('result', {}).get('data_array', [])

            for doc in docs:
                if doc[-1] > 0.5:
                    image_path = f"/Volumes/dev/haruna_osaki/images/{doc[4]}"
                    returned_docs.append({
                        "id": doc[0],
                        "detail": doc[1],
                        "category": doc[2],
                        "color": doc[3],
                        "image_path": image_path
                    })

            return returned_docs

        except requests.exceptions.RequestException as e:
            print(f"リクエストエラー: {e}")
        except Exception as e:
            if "CANCELLED" in str(e) and attempt < max_retries - 1:
                print(f"モデルがスケールアップ中… {wait_time}秒待機 (試行 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Vector search error: {e}")
                return []

def generate_and_find_coordinations(body_type_result, color_type_result):
    """コーディネート提案と類似コーディネーション検索を行う関数"""
    try:
        # コーディネート提案を生成
        coordination_text = generate_coordination(body_type_result, color_type_result)
        print(f"Generated coordination text: {coordination_text}")  # デバッグ用
        
        # 類似のコーディネーションを検索
        similar_coordinations = find_similar_coordinations(coordination_text)
        print(f"Found similar coordinations: {similar_coordinations}")  # デバッグ用
        
        # 画像パスのリストを作成
        similar_images = []
        for doc in similar_coordinations:
            if 'image_path' in doc and doc['image_path']:
                similar_images.append(doc['image_path'])
        
        print(f"Final image paths: {similar_images}")  # デバッグ用
        return coordination_text, similar_images
    
    except Exception as e:
        print(f"コーディネーション生成中にエラーが発生しました: {e}")
        return "エラーが発生しました", []
    


def diagnose_body_type(image):
    """骨格診断を行う関数"""
    if image is None:
        return "画像が提供されていません"
    
    base64_image = encode_image(image)
    
    query = """
    あなたはファッションコンサルタントとして、ユーザーの骨格診断を正確に行います。
    以下の詳細な基準に従い、診断を実施してください。

    ### 診断手順

    1. **画像を解析し、骨格の特徴を評価してください。**
    - **骨のフレーム:** （しっかりした骨格か、華奢な骨格か）
    - **筋肉の付き方:** （ハリがあるか、ソフトか）
    - **脂肪の付き方:** （均等か、一部に偏りがあるか）
    - **体の重心:** （上半身寄りか、下半身寄りか）

    2. **以下の12分類に基づいて診断してください。**
    - **骨格ストレート:** （ザ・ストレート / スレンダー・ストレート / ソフト・ストレート / ラフ・ストレート）
    - **骨格ウェーブ:** （ザ・ウェーブ / メリハリ・ウェーブ / リッチ・ウェーブ / ラフ・ウェーブ）
    - **骨格ナチュラル:** （ザ・ナチュラル / メリハリ・ナチュラル / リッチ・ナチュラル / ラフ・ナチュラル）

    3. **特徴を詳細に記述し、診断結果を以下の形式で返してください。**
    - **タイプ:** [診断された骨格タイプ]
    - **特徴:** [3〜4行の詳細な特徴説明]
    - **おすすめのスタイル:** [その骨格タイプに最適な服装やコーディネートのヒント]
    - **避けるべきスタイル:** [似合いにくい服の特徴とその理由]

    ### 診断基準の詳細

    #### **ストレートタイプ:**
    - 体に厚みがあり、筋肉のハリが強い。上重心で立体的なシルエット。
    - **似合う服:** 直線的でシンプルなデザイン、ハリのある素材。
    - **避ける服:** フリルや装飾が多すぎるデザイン、ゆるすぎる服。

    #### **ウェーブタイプ:**
    - 華奢で柔らかい質感、脂肪がつきやすく曲線的なライン。下重心。
    - **似合う服:** 軽やかでソフトな素材、フィット＆フレアのシルエット。
    - **避ける服:** ボクシーなシルエット、硬い素材。

    #### **ナチュラルタイプ:**
    - 骨の存在感があり、フレームがしっかりしている。重心はバランスが良い。
    - **似合う服:** ラフで無造作なデザイン、オーバーサイズの服。
    - **避ける服:** タイトすぎる服、光沢のある繊細な素材。

    一貫性を保つため、上記の判断基準を厳密に適用してください。
    """
    
    response = client.chat.completions.create(
        model="aoai-gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは専門的な骨格診断ができるファッションコンサルタントです。"},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=300,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def diagnose_personal_color(image):
    """パーソナルカラー診断を行う関数"""
    if image is None:
        return "画像が提供されていません"
    
    base64_image = encode_image(image)
    
    query = """
    あなたは専門的なパーソナルカラー診断を行うAIアシスタントです。以下の手順に従い、正確な診断を実施してください。

    ## 診断手順
    1. 入力された画像を解析し、以下の4つのパーソナルカラータイプのいずれかを選定してください。
        - **スプリング（イエベ春）**: 明るく鮮やかな暖色。黄味のある肌。
        - **サマー（ブルベ夏）**: 柔らかく優しい寒色。青味のある肌。
        - **オータム（イエベ秋）**: 深みのある暖色。黄味がかった肌。
        - **ウィンター（ブルベ冬）**: 鮮やかで濃い寒色。青白い肌。

    2. 以下の特徴をもとに判断してください。
        - **肌の色**: 黄味が強いか、青味が強いかを分析。
        - **髪の色**: 黒髪、茶髪、赤み、黄みの有無を考慮。
        - **目の色**: 明るさ、透明感、深みの有無を観察。

    3. 診断は必ず上記の基準に基づいて行い、ランダムな判断は避けてください。

    ## 診断結果のフォーマット
    診断結果は以下の形式で出力してください。

    \"\"\"
    タイプ: [診断されたシーズンタイプ]
    特徴: [肌、髪、目の色の特徴]
    似合う色: [そのタイプに最適な色の例]
    避けるべき色: [そのタイプに合わない色の例]
    \"\"\"

    ## 各タイプの詳細

    ### スプリング（イエベ春）
    - **特徴**: 明るく鮮やかな暖色が似合う。肌は黄みがあり、血色感がある。
    - **似合う色**: コーラル、ピーチピンク、ビタミンオレンジ、ネープルスイエロー、レタスグリーン、キャンディーブルー
    - **避けるべき色**: スモーキーな色、暗くくすんだ色

    ### サマー（ブルベ夏）
    - **特徴**: 柔らかく優しい寒色が似合う。肌は青みがあり、透明感がある。
    - **似合う色**: ローズピンク、レモンイエロー、セージグリーン、サックスブルー、ラベンダー
    - **避けるべき色**: 鮮やかすぎる色、黄みの強い色

    ### オータム（イエベ秋）
    - **特徴**: 深みのある暖色が似合う。肌は黄みが強く、落ち着いた印象。
    - **似合う色**: サーモン、アプリコット、マスタード、テラコッタ、カーキ、ターコイズブルー
    - **避けるべき色**: パステルカラー、冷たい青みの強い色

    ### ウィンター（ブルベ冬）
    - **特徴**: 鮮やかでコントラストの強い寒色が似合う。肌は青白く、クールな印象。
    - **似合う色**: バーガンディ、フューシャピンク、レモンイエロー、ビリヤードグリーン、ロイヤルブルー、ロイヤルパープル
    - **避けるべき色**: 黄みの強い色、くすんだ色

    必ずこのルールに従い、適切な診断を行ってください。
    """
    response = client.chat.completions.create(
        model="aoai-gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは専門的なパーソナルカラー診断ができるスタイリストです。"},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=300,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def generate_coordination(body_type_result, color_type_result):
    """コーディネート提案を生成する関数"""
    query = f"""以下の診断結果に基づいて、最適なコーディネートを提案してください：
    
    {body_type_result}
    
    {color_type_result}

    提案は以下の形式で返してください：
    トップス: [具体的な服のアイテム]
    ボトムス: [具体的な服のアイテム]
    アウター/羽織: [必要に応じて]
    小物: [アクセサリーやバッグなど]
    """
    
    response = client.chat.completions.create(
        model="aoai-gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは最先端のファッションスタイリストです。"},
            {"role": "user", "content": query}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def main_app():
    """メインのGradioアプリケーション"""
    with gr.Blocks(css="""
        /* メインコンテンツのスペーシング */
        .main-content {
            margin-bottom: 20px;  /* 余分なスペースを減らす */
        }

        /* ツールチップコンテナ */
        .terms-container {
            margin-top: 20px;  /* 上部のスペースを減らす */
            margin-bottom: 20px;  /* 下部のスペースを減らす */
            position: relative;
        }

        /* カスタムツールチップ */
        .custom-tooltip {
            display: inline-block;
            border-bottom: 1px dotted #000;
            position: relative;
            cursor: pointer;
            margin: 0 4px;
        }

        .custom-tooltip .tooltip-text {
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 12px;
            width: 250px;
            
            /* ポジショニング調整 */
            position: fixed;  /* fixedに変更してページ全体に対して配置 */
            z-index: 10000;  /* より高いz-indexを設定 */
            transform: translateX(-50%);
            
            /* ツールチップの位置を動的に計算 */
            left: 50%;
            bottom: auto;  /* 自動位置調整 */
            
            /* 表示制御 */
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s, visibility 0.3s;
            
            /* テキスト設定 */
            font-size: 14px;
            line-height: 1.5;
            white-space: normal;
            
            /* 背景をより目立たせる */
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        /* ホバー時の表示 */
        .custom-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
            /* ホバー時にツールチップを要素の上に配置 */
            transform: translate(-50%, -100%);
            margin-bottom: 15px;  /* 矢印のためのスペース */
        }

        /* 矢印 */
        .custom-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
    """) as demo:
        # マークダウンをdivで囲む
        gr.HTML("""
        <div class="main-content">
            <h1>CHIC</h1>
            <p>トレンドと科学で導く、あなたのパーソナルスタイル</p>
            
            <h3>使い方</h3>
            <ol>
                <li>全身写真をアップロードしてください</li>
                <li>「診断開始」を押すと、あなたの骨格タイプとベストカラーを分析します</li>
                <li>「コーディネート提案」で、あなたに似合うスタイルをご提案します</li>
            </ol>
        </div>
        """)
        
        # 専門用語の説明
        gr.HTML("""
        <div class="terms-container">
            <span class="custom-tooltip">
                骨格診断
                <span class="tooltip-text">体の骨格や筋肉のつき方から、あなたに似合うファッションスタイルを見つける診断方法です</span>
            </span>
            ・
            <span class="custom-tooltip">
                パーソナルカラー診断
                <span class="tooltip-text">肌の色や髪の色から、あなたに最も似合う色を見つける診断方法です</span>
            </span>
            について詳しく知りたい方は各用語にカーソルを合わせてください。
        </div>
        """)
        
        # 画像アップロード
        with gr.Column():
            gr.Markdown("""
            📸 **写真のポイント**
            - 全身が写っている写真を使用してください
            - なるべく体のラインが分かりやすい服装で
            - 自然光の下で撮影すると、より正確な診断が可能です
            """)
        upload_input = gr.Image(
            type="filepath", 
            label="あなたの写真をアップロード",
            height=700,  
            width=1000,   
            )
        
        with gr.Row():
            combined_diagnosis_btn = gr.Button("診断開始")
        
        # 診断結果出力
        body_diagnosis_output = gr.Textbox(label="骨格診断結果")
        color_diagnosis_output = gr.Textbox(label="パーソナルカラー診断結果")
        
        # コーディネート提案ボタン
        coordination_btn = gr.Button("コーディネート提案")
        coordination_output = gr.Textbox(label="コーディネート提案")

        # 類似コーディネーション表示用のギャラリー
        similar_images = [
            "stylist_ai/001.jpg.avif",
            "stylist_ai/002.jpg.avif"
        ]
        similar_coordinations_gallery = gr.Gallery(
            #value = process_coordinations,
            label="おすすめのコーディネーション", 
            show_label=True,
            elem_id="coordination-gallery",
            columns=2,
            object_fit="contain",
            height="auto",
            allow_preview=True,
            type="filepath",
            format="PNG"      # フォーマットを明示的に指定
        )
        
        # 診断ボタンのイベントハンドラ
        def combined_diagnosis(image):
            """骨格診断とパーソナルカラー診断を同時に行う関数"""
            body_result = diagnose_body_type(image)
            color_result = diagnose_personal_color(image)
            return body_result, color_result
        
        def process_coordinations(body_result, color_result):
            coordination_text, similar_coordinations = generate_and_find_coordinations(body_result, color_result)
            
            # 画像パスのリストを作成
            image_paths = similar_coordinations

            print(image_paths)

            
            return coordination_text, image_paths
            
            
        
        # イベントハンドラの定義
        combined_diagnosis_btn.click(
            fn=combined_diagnosis, 
            inputs=upload_input, 
            outputs=[body_diagnosis_output, color_diagnosis_output]
        )
        
        # コーディネート提案と類似画像検索のイベント
        coordination_btn.click(
            fn=process_coordinations,
            inputs=[body_diagnosis_output, color_diagnosis_output],
            outputs=[coordination_output, similar_coordinations_gallery]
        )
    
    return demo

# アプリケーションの起動
if __name__ == "__main__":
    demo = main_app()
    demo.launch()