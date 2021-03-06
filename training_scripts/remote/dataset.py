import tensorflow as tf
import pandas as pd

# install tensorflow io here because it is not using pip in the docker image build.
# !pip uninstall tensorflow-io
# !pip install tensorflow-io==0.17.1

import tensorflow_io as tfio
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

class SimpleDataset() :
    def __init__(self, data_path, name, batch_size=32, is_test=False,
                 files_list=False, sr=32000):
        self.data_path = data_path
        self.files_list = files_list
        self.is_test = is_test
        self.batch_size = batch_size
        self.name = name
        self.sr = sr

    def get_labels_from_path(self, filename):

        label = tf.strings.split(filename, sep='/')[-2]

        label_enum = self.species_enum_table.lookup(label)

        onehot_label = tf.one_hot(label_enum,
                                 self.nspecies)


        # onehot_label = tf.reshape(onehot_label, [1, -1])
        return filename, onehot_label

    def load_audio(self, file_path, label):


        # can we load a partial file??
        audio = tf.io.read_file(file_path)
        #audio = tfio.audio.decode_vorbis(audio)
        audio, sr = tf.audio.decode_wav(audio)

        # sr = tf.constant(sr, dtype=tf.int64)
        # ssr = tf.constant(self.sr, dtype=tf.int64)
        # audio = tfio.audio.resample(audio, sr, self.sr)
        # Assume all files have been resampled to 32khz
        #sr = 32000
        seconds = 10
        samples = sr * seconds

        # crop to first 10 seconds
        audio = audio[:samples]

        # pad it out if its shorter
        back_pad = samples - tf.shape(audio)[0]
        paddings = tf.stack([[0, back_pad], [0,0]])
        audio = tf.pad(audio, paddings)

        return audio, label #, file_path

    def get_species_enum_table(self):
        # Create a static enum for the species set.

        # get rid of the pesky .DS_Store files
        if os.path.exists(os.path.join(self.data_path, '.DS_Store')):
                os.remove(os.path.join(self.data_path, '.DS_Store'))
        species_list = sorted(os.listdir(self.data_path))
        print(species_list[:5])

        self.species_list = species_list
        self.nspecies = len(species_list)
        species_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(species_list),
                                                tf.constant(range(self.nspecies))),
            default_value=-1)

        # reverse_species_table = tf.lookup.StaticHashTable(
        #     tf.lookup.KeyValueTensorInitializer(tf.constant(range(self.nspecies)),
        #                                         tf.constant(species_list)),
        #     default_value=-1)

        return species_table #, reverse_species_table

    def get_dataset(self):
        self.species_enum_table = self.get_species_enum_table()

        # if self.files_list:
        #     # if a list of file names was provided
        #     print('creating from list of length:', len(self.files_list))
        file_paths_ds = tf.data.Dataset.from_tensor_slices(self.files_list)
        # else:
        #     # if no list of filenames use glob
        #     print('creating from path')
        #     file_paths_ds = tf.data.Dataset.list_files(os.path.join(self.data_path,'*/*.wav'),
        #                                            shuffle=False)

        # add labels
        labels_ds = file_paths_ds.map(self.get_labels_from_path, num_parallel_calls=AUTOTUNE)

        # setup dataset
        if not self.is_test:
            # shuffle if not test set
            labels_ds = labels_ds.shuffle(len(self.files_list), reshuffle_each_iteration=True)


        ds = labels_ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)

        # ds.cache('./cache/'+self.name) # is this overwriting the same file?


        ds = ds.batch(self.batch_size)
        print('Batch Size set to ', self.batch_size)
        #ds = ds.batch(2)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds


class SoundScapeDataset() :
    def __init__(self, data_path, name, files_list, labels_list,
                 batch_size=32, is_test=False, sr=22050):
        self.data_path = data_path
        self.files_list = files_list
        self.labels_list = labels_list
        self.is_test = is_test
        self.batch_size = batch_size
        self.name = name
        self.sr = sr

        # Placeholders
        self.noise_paths_ds = None

    def get_labels_from_path(self, filename):

        label = tf.strings.split(filename, sep='/')[-2]

        label_enum = self.species_enum_table.lookup(label)

        onehot_label = tf.one_hot(label_enum,
                                 self.nspecies)


        # onehot_label = tf.reshape(onehot_label, [1, -1])
        return filename, onehot_label

    def load_audio(self, file_path, noise_path=False):

        # can we load a partial file??
        audio = tf.io.read_file(file_path)
        audio, sr = tf.audio.decode_wav(audio)

        seconds = 5
        samples = sr * seconds

        # crop to first 15 seconds
        audio = audio[:samples]

        # pad it out if its shorter
        back_pad = samples - tf.shape(audio)[0]
        paddings = tf.stack([[0, back_pad], [0,0]])
        audio = tf.pad(audio, paddings)

        return audio #, file_path

    def get_species_enum_table(self):
        # Create a static enum for the species set.

        # get rid of the pesky .DS_Store files
        if os.path.exists(os.path.join(self.data_path, '.DS_Store')):
                os.remove(os.path.join(self.data_path, '.DS_Store'))
        species_list = sorted(os.listdir(self.data_path))
        print(species_list[:5])

        self.species_list = species_list
        self.nspecies = len(species_list)
        species_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(species_list),
                                                tf.constant(range(self.nspecies))),
            default_value=-1)

        return species_table

    def process_examples(self, ex0, ex1=None, is_train=True):
        if ex1 is not None:
            # Here we combine two examples.
            labels = tf.stack([ex0['label'], ex1['label']],
                                   axis=1)
            labels_enum = tf.stack([self.species_enum_table.lookup(ex0['label']),
                                    self.species_enum_table.lookup(ex1['label'])],
                                   axis=1)

            # This applies a random gain to each member of the batch separately.
            gain0 = tf.random.uniform([self.batch_size, 1, 1], 0.2, 0.5)
            merged_audio = gain0 * ex0['audio'] + (1 - gain0) * ex1['audio']
        else:
            labels = ex0['label'][:, tf.newaxis]
            labels_enum = self.species_enum_table.lookup(ex0['label'])[:, tf.newaxis]
            merged_audio = ex0['audio']

    def get_dataset(self):

        file_paths_ds = tf.data.Dataset.from_tensor_slices(self.files_list)

        labels_ds = tf.data.Dataset.from_tensor_slices(self.labels_list)

        ds = tf.data.Dataset.zip((file_paths_ds, labels_ds))

        ds = ds.shuffle(len(self.files_list), reshuffle_each_iteration=True)

        ds = ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds


class CombinedDataset():
    def __init__(self, files_list, labels_list,
                noise_files=None, batch_size=32, is_train=None, sr=32000):
        self.files_list = files_list
        self.labels_list = labels_list
        self.noise_files = noise_files
        self.is_train = is_train
        self.batch_size = batch_size
        self.sr = sr

    def get_labels_from_path(self, filename):

        label = tf.strings.split(filename, sep='/')[-2]

        label_enum = self.species_enum_table.lookup(label)

        onehot_label = tf.one_hot(label_enum,
                                 self.nspecies)

        # onehot_label = tf.reshape(onehot_label, [1, -1])
        return filename, onehot_label

    def load_audio(self, file_path, label, noise_path=None):

        # can we load a partial file??
        audio = tf.io.read_file(file_path)
        audio, sr = tf.audio.decode_wav(audio)

        seconds = 5
        samples = sr * seconds

        # crop to first 15 seconds
        audio = audio[:samples]

        # pad it out if its shorter
        back_pad = samples - tf.shape(audio)[0]
        paddings = tf.stack([[0, back_pad], [0,0]])
        audio = tf.pad(audio, paddings)

        # gain0 = 0

        print(noise_path)

        if noise_path is not None:
            noise = tf.io.read_file(noise_path)
            noise, sr = tf.audio.decode_wav(noise)

            gain0 = tf.random.uniform([], 0.4, 0.6)

            audio = gain0 * noise + (1 - gain0) * audio

        else:
            noise_path = 'none'
        return audio, label #, gain0 #, file_path


    def get_dataset(self):

        file_paths_ds = tf.data.Dataset.from_tensor_slices(self.files_list)

        onehot_label = tf.one_hot(self.labels_list, 397) #n species
        labels_ds = tf.data.Dataset.from_tensor_slices(onehot_label)

        if self.is_train == True:
            # get a noise_paths_ds with the same length as files_list
            noise_paths_ds = tf.data.Dataset.from_tensor_slices(self.noise_files)
            noise_paths_ds = noise_paths_ds.repeat().shuffle(1024)

            paths_zip_ds = tf.data.Dataset.zip((file_paths_ds, labels_ds, noise_paths_ds))
            paths_zip_ds = paths_zip_ds.shuffle(len(self.files_list), reshuffle_each_iteration=True)

        else:

            paths_zip_ds = tf.data.Dataset.zip((file_paths_ds, labels_ds))

        ds = paths_zip_ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

class CombinedDatasetFullAudio():
    def __init__(self, files_list, labels_list,
                noise_files=None, batch_size=32, is_train=None, sr=32000):
        self.files_list = files_list
        self.labels_list = labels_list
        self.noise_files = noise_files
        self.is_train = is_train
        self.batch_size = batch_size
        self.sr = sr

        self.vggish = self.load_vggish()

    def load_vggish(self):
        return tf.saved_model.load('vggish_1')


    def get_labels_from_path(self, filename):

        label = tf.strings.split(filename, sep='/')[-2]

        label_enum = self.species_enum_table.lookup(label)

        onehot_label = tf.one_hot(label_enum,
                                 self.nspecies)

        # onehot_label = tf.reshape(onehot_label, [1, -1])
        return filename, onehot_label

    def load_audio(self, file_path, label, noise_path=None):

        # can we load a partial file??
        audio = tf.io.read_file(file_path)
        audio, sr = tf.audio.decode_wav(audio)

        seconds = 5
        samples = sr * seconds

        # crop to first 5 seconds
        audio = audio[:samples]

        # pad it out if its shorter
        back_pad = samples - tf.shape(audio)[0]
        paddings = tf.stack([[0, back_pad], [0,0]])
        audio = tf.pad(audio, paddings)

        # gain0 = 0

#         print(noise_path)

        if noise_path is not None:
            noise = tf.io.read_file(noise_path)
            noise, sr = tf.audio.decode_wav(noise)

            gain0 = tf.random.uniform([], 0.3, 0.6)


            audio = gain0 * noise + (1 - gain0) * audio

        else:
            noise_path = 'none'
        return audio, label #, gain0 #, file_path


    def get_vggish_embeding(self, file_path, label):
        # can we load a partial file??
        audio = tf.io.read_file(file_path)
        audio, sr = tf.audio.decode_wav(audio)
        audio = tf.reshape(audio, [-1])

        # cut to 7 seconds max
        audio = audio[:sr*7]

        if sr != 16000:
            rate_in = tf.cast(sr, tf.int64, name=None)
            rate_out = 16000

            audio = tfio.audio.resample(audio, rate_in, rate_out, name=None)

        return self.vggish(audio), label

    def get_dataset(self):

        file_paths_ds = tf.data.Dataset.from_tensor_slices(self.files_list)

        onehot_label = tf.one_hot(self.labels_list, 397) #n species
        labels_ds = tf.data.Dataset.from_tensor_slices(onehot_label)

        if self.is_train == True:
            # # get a noise_paths_ds with the same length as files_list
            # noise_paths_ds = tf.data.Dataset.from_tensor_slices(self.noise_files)
            # noise_paths_ds = noise_paths_ds.repeat().shuffle(1024)

            paths_zip_ds = tf.data.Dataset.zip((file_paths_ds, labels_ds)) #, noise_paths_ds))
            paths_zip_ds = paths_zip_ds.shuffle(len(self.files_list), reshuffle_each_iteration=True)

        else:

            paths_zip_ds = tf.data.Dataset.zip((file_paths_ds, labels_ds))

        #ds = paths_zip_ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)
        ds = paths_zip_ds.map(self.get_vggish_embeding, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

# if __name__ == '__main__':
#     print("Testing SoundScapeDataset")
#     print('-'*100)
#
#
#     DATA_PATH = '../../sswav5sec22k'
#     BATCH_SIZE= 2
#
#     df = pd.read_pickle('../../sswav5sec22k/soundscapes.pkl')
#     files = df['row_id'].map(lambda x: os.path.join(DATA_PATH, x+'.wav')).tolist()
#     train_labels = df['label'].tolist()
#
#     # print(train_files)
#
#     sd = SoundScapeDataset('../../sswav5sec22k',
#                        name='train',
#                        batch_size=BATCH_SIZE,
#                        files_list=files,
#                        labels_list=train_labels,
#                        is_test=True)
#
#
#     ds = sd.get_dataset()
#
#     DATASET_SIZE = len(files)
#
#     train_size = int(0.9 * DATASET_SIZE / BATCH_SIZE)
#     val_size = int(0.1 * DATASET_SIZE / BATCH_SIZE)
#
#     train_ds = ds.take(train_size)
#     val_ds = ds.skip(train_size)
#
#     train_iter = train_ds.as_numpy_iterator()
#     val_ds = val_ds.as_numpy_iterator()
#
#     print('train', next(train_iter))
#     print('validation', next(val_ds))
#     print('printed iteration')
if __name__ == '__main__':
    print("Testing SimpleDataset")
    print('-'*100)

    # imports for testing
    import json
    import numpy as np

    BATCH_SIZE=2
    # glob
    # sd = SimpleDataset('../../input/birdclef-2021/train_short_audio')

    # create dataset with list
    # load the data ids
    label_file = 'data_split_single_label_tv.json'
    data_path = '../../wav'

    with open(os.path.join(data_path, 'input/resources', label_file), 'r') as f:
        data =  json.load(f)

#     # get file lists for train, val, test
    train_files = [os.path.join(data_path, name)
               for name in data['train']['files']]
    val_files = [os.path.join(data_path, name)
               for name in data['val']['files']]

    classes = data['mapping']
    n_classes = len(classes)

    # print('classes: ', classes)
    from sklearn.preprocessing import LabelEncoder

    le =LabelEncoder()
    le.fit(classes)

    # transform labels
    train_labels = le.transform(data['train']['labels']).tolist()
    val_labels = le.transform(data['val']['labels']).tolist()

    # select files from soundscapes without birds.
    ssdf = pd.read_csv('../../input/birdclef-2021/train_soundscape_labels.csv')
    ssdf_filtered = ssdf[ssdf['birds'] == 'nocall']
    noise_files = [os.path.join('../../sswav5sec', fid + '.wav') for fid in ssdf_filtered['row_id']]

    # number of steps to test
    steps = np.floor(len(train_files)/BATCH_SIZE)


    cd = CombinedDataset(train_files, train_labels, noise_files, batch_size=1, is_train=True)
    ds = cd.get_dataset()

    ds_iter = ds.as_numpy_iterator()
    audio, label = next(ds_iter)

    print(audio)
    print(label)

    print('Dataset Cardinality:', tf.data.experimental.cardinality(ds).numpy())
    print('Noise Files:', len(noise_files))

#
#
#
#     print('Creating validation DataGenerator...')
#
#     from sklearn.preprocessing import LabelEncoder
#     classes = data['mapping']
#     n_classes = len(classes)
#
#
#     le =LabelEncoder()
#     le.fit(classes)
#
#     sr=32000
#     dt=10
#
#     from DataGenerator import DataGenerator
#
#     train_labels = le.transform(data['train']['labels']).tolist()
#     train_generator = DataGenerator(wav_paths=train_files, #[:32],
#                                     labels=train_labels, #[:32],
#                                     sr=sr,
#                                     dt=dt,
#                                     n_classes=len(classes),
#                                     batch_size=BATCH_SIZE,
#                                     shuffle=False)
#
#
#     ds_iter = ds.as_numpy_iterator()
#
#     matches = 0
#
#     for step in range(int(steps)):
#
#         audio, label = next(ds_iter)
#         audiogen, labelgen = train_generator.__getitem__(step)
#
#         if np.array_equal(audio, audiogen) and np.array_equal(label, labelgen):
#             print(f'Step: {step}/{int(steps)}, Audio: True')
#             matches += 1
#         # else:
#         #     print(f'Step: {step}/{int(steps)}, Audio: False')
#         #
#         #     for idx in range(len(label)):
#         #         print('------> Item', idx)
#         #
#         #         print('file ds', filepath[idx])
#         #         print('file generator', filepathgen[idx])
#         #
#         #         print('audio ds', audio[idx].shape, 'sum', np.sum(audio[idx]), audio[idx])
#         #         print('audio generator', audiogen[idx].shape, 'sum', np.sum(audiogen[idx]), audiogen[idx])
#         #         test = np.array_equal(audio[idx], audiogen[idx])
#         #         if not test:
#         #             print(test)
#         #             break
#         #
#         #         print('label ds', label[idx])
#         #         print('label generator', labelgen[idx])
#         #
#         #         test = np.array_equal(label[idx], labelgen[idx])
#         #         if not test:
#         #             print(test)
#         #             break
#         #     break
#
#     print(f'{matches}/{steps} steps were equal')
#
#     #
#     #
#     #
#     # for idx in range(len(label)):
#     #     print('------> Item', idx)
#     #     print('audio ds', audio[idx].shape, audio[idx])
#     #     print('audio generator', audiogen[idx].shape, audiogen[idx])
#     #
#     #
#     #     print('label ds', label[idx])
#     #     print('label generator', labelgen[idx])
#     #
#     #     print(np.array_equal(label[idx], labelgen[idx]))
#     #
#     #     print('file ds', filepath[idx])
#     #     print('file generator', filepathgen[idx])
#
#
#
# #
# # #    print(sd.species_enum_table.lookup([b'acafly']))
#
# #     model = tf.keras.Sequential([
# #         tf.keras.layers.Flatten(input_shape=(320000, 1)),
# #         tf.keras.layers.Dense(397)
# #     ])
# #
# #     model.summary()
# #
# #     model.compile(optimizer='adam',
# #         loss=tf.keras.losses.CategoricalCrossentropy(),
# #         metrics=['accuracy'])
# #
# # #     model.summary()
# # #
#     # model.fit(ds, epochs=2)
