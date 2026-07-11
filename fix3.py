with open('index.html', 'r') as f:
    content = f.read()

old = """      const tensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(video)  // RGB [H, W, 3]
        const gray = img.mean(2, true)             // grayscale [H, W, 1]
        const resized = tf.image.resizeBilinear(gray, [48, 48])
        const normalized = resized.div(255.0)
        return normalized.expandDims(0)            // [1, 48, 48, 1]
      });"""

new = """      const tensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(video)
        // Conversion RGB -> grayscale comme OpenCV
        // 0.299*R + 0.587*G + 0.114*B
        const r = img.slice([0,0,0], [-1,-1,1]).mul(0.299)
        const g = img.slice([0,0,1], [-1,-1,1]).mul(0.587)
        const b = img.slice([0,0,2], [-1,-1,1]).mul(0.114)
        const gray = r.add(g).add(b)
        const resized = tf.image.resizeBilinear(gray, [48, 48])
        const normalized = resized.div(255.0)
        return normalized.expandDims(0)
      });"""

content = content.replace(old, new)
with open('index.html', 'w') as f:
    f.write(content)
print("Done!")
