import tensorflow as tf

# Create two constants: a and b
a = tf.constant(4)
b = tf.constant(3)

# Perform a computation
c = a + b

# In TensorFlow 2.x, eager execution is enabled by default,
# so you can directly inspect the result.
print(c.numpy())  # This should print 7
