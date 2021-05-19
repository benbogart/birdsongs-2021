import tensorflow as tf
import tensorflow_io as tfio
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

class SimpleDataset() :
    def __init__(self, data_path, is_test=False, files_list=False):
        self.data_path = data_path
        self.files_list = files_list
        self.is_test = is_test

    def get_labels_from_path(self, filename):

        label = tf.strings.split(filename, sep='/')[-2]

        label_enum = self.species_enum_table.lookup(label)

        onehot_label = tf.one_hot(label_enum,
                                 self.nspecies)

        # onehot_label = tf.reshape(onehot_label, [1, -1])
        return filename, onehot_label

    def load_audio(self, file_path, label):

        # Assume all files have been resampled to 32khz
        sr = 32000
        seconds = 10
        samples = sr * seconds
        # can we load a partial file??
        audio = tf.io.read_file(file_path)
        audio = tfio.audio.decode_vorbis(audio)

        # crop to first 10 seconds
        audio = audio[:samples - 1000] # to test pading


        # pad it out if its shorter
        back_pad = samples - tf.shape(audio)[0]
        paddings = tf.stack([[0, back_pad], [0,0]])
        audio = tf.pad(audio, paddings)

        return audio, label,

    def get_species_enum_table(self):
        # Create a static enum for the species set.

        # get rid of the pesky .DS_Store files
        if os.path.exists(os.path.join(self.data_path, '.DS_Store')):
                os.remove(os.path.join(self.data_path, '.DS_Store'))
        species_list = sorted(os.listdir(self.data_path))
        print(species_list[:5])

        self.nspecies = len(species_list)
        species_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(species_list),
                                                tf.constant(range(self.nspecies))),
            default_value=-1)
        return species_table

    def get_dataset(self):
        self.species_enum_table = self.get_species_enum_table()

        if self.files_list:
            # if a list of file names was provided
            file_paths_ds = tf.data.Dataset.from_tensor_slices(self.files_list)
        else:
            # if no list of filenames use glob
            file_paths_ds = tf.data.Dataset.list_files(os.path.join(self.data_path,'*/*.ogg'),
                                                   shuffle=False)


        labels_ds = file_paths_ds.map(self.get_labels_from_path, num_parallel_calls=AUTOTUNE)
        ds = labels_ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)



        # if not self.is_test:
        #     # shuffle and repeat indefently
        #     ds = ds.shuffle(1024, reshuffle_each_iteration=True)
        #     ds = ds.repeat()

        return ds

if __name__ == '__main__':
    print("Testing SimpleDataset")
    print('-'*100)

    # imports for testing
    import json

    # glob
    # sd = SimpleDataset('../../input/birdclef-2021/train_short_audio')

    # create dataset with list
    # load the data ids
    label_file = 'data_split_single_label.json'
    data_path = '../../'

    with open(os.path.join(data_path, 'input/resources', label_file), 'r') as f:
        data =  json.load(f)

    # get file lists for train, val, test
    train_files = [os.path.join(data_path, name)
               for name in data['train']['files']]
    val_files = [os.path.join(data_path, name)
               for name in data['val']['files']]
    test_files = [os.path.join(data_path, name)
               for name in data['test']['files']]


    print(train_files[:3])

    sd = SimpleDataset('../../input/birdclef-2021/train_short_audio', train_files)

    ds = sd.get_dataset()

    ds = ds.batch(3)
    ds.as_numpy_iterator()


    r = 0
    for audio, label in ds.as_numpy_iterator():
        print('audio shape:', audio.shape)
        print('label shape:', label.shape)
        r += 1
        #print(row[0].shape)
        if r > 9:
            break
#
# #    print(sd.species_enum_table.lookup([b'acafly']))

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(320000, 1)),
        tf.keras.layers.Dense(397)
    ])

    model.summary()

    model.compile(optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

#     model.summary()
#
    model.fit(ds, epochs=2)
