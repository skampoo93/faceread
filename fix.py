with open('index.html', 'r') as f:
    content = f.read()

old = """      const preds = await model.predict(tensor).data();
      tensor.dispose();"""

new = """      const output = model.predict(tensor);
      const preds = await output.data();
      tensor.dispose();
      output.dispose();"""

content = content.replace(old, new)

with open('index.html', 'w') as f:
    f.write(content)
print("Done!")
