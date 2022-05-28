import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras.layers as L
from matplotlib import pyplot as plt
import time

(train_set, _), ds_info = tfds.load('mnist', split=[
    'train', 'test'], as_supervised=True, shuffle_files=True, with_info=True)

# for (e, l) in train_set:
#     pixels = np.array(e, dtype='uint8')
#     pixels.reshape((28, 28))
#     plt.imshow(pixels, cmap='gray')
#     plt.show()
#     break


def normalize_img(image):
    fin = tf.cast(image, tf.float32)
    return (fin - 127.5) / 127.5


slices = []
for image, _ in train_set:
    slices.append(normalize_img(image))
train_images = tf.data.Dataset.from_tensor_slices(slices)
train_images = train_images.cache()
train_images = train_images.shuffle(ds_info.splits['test'].num_examples)
train_images = train_images.batch(256)
train_images = train_images.prefetch(tf.data.AUTOTUNE)


discriminator = keras.models.Sequential()
discriminator.add(L.Conv2D(64, 5, 2, 'same', input_shape=(28, 28, 1)))
discriminator.add(L.LeakyReLU())
discriminator.add(L.Dropout(0.3))
discriminator.add(L.Conv2D(128, 5, 2, 'same'))
discriminator.add(L.LeakyReLU())
discriminator.add(L.Dropout(0.3))
discriminator.add(L.Flatten())
discriminator.add(L.Dense(1))

generator = keras.models.Sequential()
generator.add(L.Dense(7*7*256, use_bias=False, input_shape=(100, )))
generator.add(L.BatchNormalization())
generator.add(L.LeakyReLU())
generator.add(L.Reshape((7, 7, 256)))
generator.add(L.Conv2DTranspose(128, 5, 1, padding='same', use_bias=False))
generator.add(L.BatchNormalization())
generator.add(L.LeakyReLU())
generator.add(L.Conv2DTranspose(64, 5, 2, padding='same', use_bias=False))
generator.add(L.BatchNormalization())
generator.add(L.LeakyReLU())
generator.add(L.Conv2DTranspose(1, 5, 2, padding='same',
              use_bias=False, activation='tanh'))

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, fake):
    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


def generator_loss(fake):
    gen_loss = loss(tf.ones_like(fake), fake)
    return gen_loss


generator_optimizer = keras.optimizers.Adam(0.0002, 0.5)
discriminator_optimizer = keras.optimizers.Adam(0.0002, 0.5)


@tf.function
def train_step(images):
    noise = tf.random.normal([256, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gen_gradients = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gen_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(disc_gradients, discriminator.trainable_variables))

        return gen_loss, disc_loss


def train(dataset, epochs):
    gen_loss, disc_loss = 0, 0
    gen_plot, disc_plot = [], []
    t = time.time()
    for epoch in range(epochs):
        print(
            f'Epoch {epoch}: generator loss -> {gen_loss}, discriminator_loss -> {disc_loss}, time elapsed: {time.time() - t}')
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        gen_plot.append(gen_loss)
        disc_plot.append(disc_loss)

    plt.title('Epoch vs Loss')
    plt.legend(['gen_loss', 'dist_loss'])
    plt.xlabel('epochs'), plt.ylabel('loss')
    x = [i+1 for i in range(epochs)]
    plt.plot(x, gen_plot), plt.plot(x, disc_plot)
    plt.show()


train(train_images, 25)

cnt = 50
noise = tf.random.normal([cnt, 100])
pred = generator.predict(noise)

fig = plt.figure(figsize=(8, 8))
for i in range(cnt):
    fig.add_subplot(cnt//5, 5, i + 1)
    plt.imshow(pred[i], cmap='gray')
plt.show()
