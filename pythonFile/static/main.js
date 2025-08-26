// const ws = new WebSocket('ws://' + location.host + '/ws');
// ws.binaryType = 'blob';
// ws.onmessage = (ev) => {
//     if (ev.data instanceof Blob) {
//         const reader = new FileReader();
//         reader.onload = () => {
//             try {
//                 // Tìm vị trí của dấu phân cách "|"
//                 const separatorIndex = reader.result.indexOf('|');
//                 if (separatorIndex === -1) {
//                     console.error('No separator found in WebSocket data');
//                     return;
//                 }
//                 // Tách metadata và frame
//                 const metadataStr = reader.result.slice(0, separatorIndex);
//                 const frameData = ev.data.slice(separatorIndex + 1);
//                 const data = JSON.parse(metadataStr);
//                 const img = document.getElementById('video_stream_' + data.stream_id);
//                 if (img) {
//                     const url = URL.createObjectURL(frameData);
//                     img.onload = () => URL.revokeObjectURL(url);
//                     img.src = url;
//                 } else {
//                     console.error('Image element not found for stream_id: ' + data.stream_id);
//                 }
//             } catch (e) {
//                 console.error('Error processing WebSocket data:', e);
//             }
//         };
//         reader.readAsText(new Blob([ev.data.slice(0, 400)])); // Tăng kích thước để đảm bảo đọc hết metadata
//     }
// };
// ws.onerror = (e) => console.error('WebSocket error:', e);
// ws.onclose = () => console.log('WebSocket closed');

// Thêm một "lớp vỏ" sự kiện bên ngoài toàn bộ code
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