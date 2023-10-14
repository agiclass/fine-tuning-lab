#!/bin/bash

REPO_DIR="/root/autodl-tmp/chatglm2-6b-official-repo" # chatglm-6b代码所在目录

if [ -d "$REPO_DIR" ]; then
    echo "Directory exists. Proceeding with running the web demo."
else
    echo "Directory does not exist. Cloning from repository."
    git clone https://github.com/THUDM/ChatGLM2-6B "$REPO_DIR"
    mkdir "$REPO_DIR/THUDM"
    ln -s "/root/autodl-tmp/chatglm2-6b" "$REPO_DIR/THUDM/chatglm2-6b"
fi

sed -i '$ s/demo.queue().launch(share=False, inbrowser=True)/demo.queue().launch(share=False, server_name='"'"'0.0.0.0'"'"', server_port=6006, inbrowser=True)/' "$REPO_DIR/web_demo.py"

cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0 python3 web_demo.py
cd -
