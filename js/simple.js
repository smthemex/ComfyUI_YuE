const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "YUESMExtension",
    nodeCreated(node) {
        if (node.comfyClass === "YUE_SM_Sampler") {
            // 创建按钮
            const button = document.createElement("button");
            button.textContent = "Select abc file upload";
            button.style.cssText = `
                margin-top: 10px;
                padding: 0 10px;
                background: #222222;
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                color: white;
                cursor: pointer;
                width: 100%;
                height: 30px;
                line-height: 30px;
                text-align: center;
                text-overflow: ellipsis;
                overflow: hidden;
                white-space: nowrap;
                box-sizing: border-box;
            `;
            
            /// 创建文件选择输入
            const input = document.createElement('input');
            input.type = 'file';
            input.style.display = 'none';
            
            input.accept = '.abc';
           
            input.multiple = false;
      
            input.nwdirectory = false;

            input.onchange = async e => {
                const file = e.target.files[0];
                if (file) {
                    const fileName = file.name;
                    //button.textContent = fileName;
                    
                    const reader = new FileReader();
                    // 辅助函数：将 ArrayBuffer 转换为 Base64 字符串
                    const arrayBufferToBase64 = (buffer) => {
                        let binary = '';
                        const bytes = new Uint8Array(buffer);
                        const len = bytes.byteLength;
                        for (let i = 0; i < len; i++) {
                            binary += String.fromCharCode(bytes[i]);
                        }
                        return window.btoa(binary);
                    };
                    reader.onload = async (event) => {
                        try {
                            // 将读取到的 ArrayBuffer 转换为 Base64
                            const fileContent = arrayBufferToBase64(event.target.result);
                            
                            const response = await fetch('/yue_sm/get_file_path', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    filename: fileName,
                                    content: fileContent,
                                    node_id: node.id
                                })
                            });
                            
                            const data = await response.json();
                            console.log('API response:', data);
                            
                            // 将后端返回的绝对路径写入到节点的 prompt_files 控件中
                            const promptFilesWidget = node.widgets.find(w => w.name === 'abc_file');
                            if (promptFilesWidget) {
                                promptFilesWidget.value = data.path;
                                // 触发回调，确保ComfyUI内部状态更新（如标记工作流为未保存）
                                if (promptFilesWidget.callback) {
                                    promptFilesWidget.callback(data.path);
                                }
                            }
                        } catch (error) {
                            console.error('Error sending file to backend:', error);
                        }
                    };
                    reader.readAsArrayBuffer(file);
                }
            };

            
            // 按钮点击事件触发文件选择
            button.onclick = () => {
                input.click();
            };
            
            // 添加DOM widget
            node.addDOMWidget("path-button", "button", button, {
                serialize: false,
                hideOnZoom: false
            });
            
            // 初始添加一个输入框用于存储路径
            // node.addInput('selected_path', 'STRING', {
            //     default: '',
            //     multiline: false,
            //     tooltip: 'Selected file path'
            // });
        }
    }
});
