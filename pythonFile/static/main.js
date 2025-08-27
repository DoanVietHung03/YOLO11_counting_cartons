document.addEventListener('DOMContentLoaded', function() {
    const videoContainer = document.querySelector('.video-container');
    if (!videoContainer) {
        console.error("Video container not found!");
        return;
    }
    
    const stream_count = parseInt(videoContainer.dataset.streamCount, 10);
    const contexts = new Map();

    // Hàm để thiết lập một kết nối WebSocket cho một stream cụ thể
    const setupWebSocket = (stream_id) => {
        const canvas = document.getElementById('video_stream_' + stream_id);
        if (!canvas) {
            console.error(`Canvas with id 'video_stream_${stream_id}' not found!`);
            return;
        }
        
        const ctx = canvas.getContext('2d', { alpha: false });
        contexts.set(stream_id, ctx);
        
        // Mỗi stream có một WebSocket riêng
        const ws = new WebSocket(`ws://${location.host}/ws/${stream_id}`);
        ws.binaryType = 'blob';

        ws.onmessage = async (ev) => {
            try {
                // Không cần đọc header stream_id nữa
                const imageBlob = ev.data;
                const bitmap = await createImageBitmap(imageBlob);

                if (ctx.canvas.width !== bitmap.width || ctx.canvas.height !== bitmap.height) {
                    ctx.canvas.width = bitmap.width;
                    ctx.canvas.height = bitmap.height;
                }
                ctx.drawImage(bitmap, 0, 0);
                bitmap.close();
            } catch (e) {
                if (!(e instanceof DOMException)) {
                     console.error(`Error processing frame for stream ${stream_id}:`, e);
                }
            }
        };

        ws.onerror = (e) => console.error(`WebSocket error on stream ${stream_id}:`, e);
        ws.onclose = () => {
            console.log(`WebSocket for stream ${stream_id} closed. Reconnecting...`);
            // Thêm logic tự động kết nối lại nếu muốn
            setTimeout(() => setupWebSocket(stream_id), 2000); // Thử kết nối lại sau 2s
        };
    };

    // Tạo kết nối cho mỗi stream
    for (let i = 0; i < stream_count; i++) {
        setupWebSocket(i);
    }
});