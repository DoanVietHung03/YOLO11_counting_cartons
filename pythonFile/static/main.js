document.addEventListener('DOMContentLoaded', function() {

    // --- Toàn bộ code cũ của bạn nằm ở trong này ---
    const videoContainer = document.querySelector('.video-container');
    // Thêm một kiểm tra để chắc chắn videoContainer tồn tại
    if (!videoContainer) {
        console.error("Video container not found!");
        return;
    }
    
    const stream_count = parseInt(videoContainer.dataset.streamCount, 10);
    const contexts = new Map();

    for (let i = 0; i < stream_count; i++) {
        const canvas = document.getElementById('video_stream_' + i);
        if (canvas) {
            // Dòng getContext bây giờ sẽ an toàn
            contexts.set(i, canvas.getContext('2d', { alpha: false }));
        } else {
            console.error(`Canvas with id 'video_stream_${i}' not found!`);
        }
    }

    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.binaryType = 'blob';

    ws.onmessage = async (ev) => {
        try {
            const data = ev.data;
            const header = await data.slice(0, 1).arrayBuffer();
            const stream_id = new Uint8Array(header)[0];
            const imageBlob = data.slice(1);
            const ctx = contexts.get(stream_id);
            if (!ctx) return;
            const bitmap = await createImageBitmap(imageBlob);
            if (ctx.canvas.width !== bitmap.width || ctx.canvas.height !== bitmap.height) {
                ctx.canvas.width = bitmap.width;
                ctx.canvas.height = bitmap.height;
            }
            ctx.drawImage(bitmap, 0, 0);
            bitmap.close();
        } catch (e) {
            if (!(e instanceof DOMException)) {
                 console.error('Error processing frame:', e);
            }
        }
    };

    ws.onerror = (e) => console.error('WebSocket error:', e);
    ws.onclose = () => console.log('WebSocket closed');
    // --- Kết thúc phần code cũ ---

});