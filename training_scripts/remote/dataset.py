import tensorflow as tf
import tensorflow_io as tfio
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

class SimpleDataset() :
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_labels_from_path(self, filename):

        label = tf.strings.split(filename, sep='/')[-2]

        label_enum = self.species_enum_table.lookup(label)

        onehot_label = tf.one_hot(label_enum,
                                 self.nspecies)
        return filename, onehot_label, label

    def load_audio(self, file_path, label, label2):

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

        return audio, label, label2

    def get_species_enum_table(self):
        # Create a static enum for the species set.

        # get rid of the pesky .DS_Store files
        if os.path.exists(os.path.join(self.file_paths, '.DS_Store')):
            os.remove(os.path.join(self.file_paths, '.DS_Store'))
        species_list = sorted(os.listdir(self.file_paths))
        print(species_list[:5])

        self.nspecies = len(species_list)
        species_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(species_list),
                                                tf.constant(range(self.nspecies))),
            default_value=-1)
        return species_table

    def get_dataset(self):
        self.species_enum_table = self.get_species_enum_table()

        file_paths_ds = tf.data.Dataset.list_files(os.path.join(self.file_paths,'*/*.ogg'),
                                                   shuffle=False)
        labels_ds = file_paths_ds.map(self.get_labels_from_path, num_parallel_calls=AUTOTUNE)
        ds = labels_ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)
        return ds

if __name__ == '__main__':
    print("Testing SimpleDataset")
    print('-'*100)

    sd = SimpleDataset('../../input/birdclef-2021/train_short_audio')

    ds = sd.get_dataset()

    r = 0
    for row in ds:
        print(row)
        r += 1
        #print(row[0].shape)
        if r > 9:
            break

    print(sd.species_enum_table.lookup(['acafly']))
