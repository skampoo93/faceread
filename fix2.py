with open('index.html', 'r') as f:
    content = f.read()

old = """      const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(video, 1)  // grayscale
          .resizeBilinear([48, 48])
          .toFloat()
          .div(255.0)
          .expandDims(0);  // [1, 48, 48, 1]
      });"""

new = """      const tensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(video)  // RGB [H, W, 3]
        const gray = img.mean(2, true)             // grayscale [H, W, 1]
        const resized = tf.image.resizeBilinear(gray, [48, 48])
        const normalized = resized.div(255.0)
        return normalized.expandDims(0)            // [1, 48, 48, 1]
      });"""

content = content.replace(old, new)
with open('index.html', 'w') as f:
    f.write(content)
print("Done!")
