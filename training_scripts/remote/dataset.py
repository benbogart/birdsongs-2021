import tensorflow as tf
import tensorflow_io as tfio

AUTOTUNE = tf.data.experimental.AUTOTUNE

class SimpleDataset() :
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_labels_from_path(self, filename):
        label = tf.strings.split(filename, sep='/')[-2]
        return filename, label

    def load_audio(self, file_path, label):
        audio = tf.io.read_file(file_path)
        audio = tfio.audio.decode_vorbis(audio)
        return audio, label

    def get_species_enum_table(self):
        # Create a static enum for the species set.
        species_list = sorted(os.listdir('/kaggle/input/birdclef-2021/train_short_audio'))
        species_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(species_list),
                                                tf.constant(range(len(species_list)))),
            default_value=-1)
        return species_table

    def get_dataset(self):
        file_paths = 'input/birdclef-2021/train_short_audio/*/*.ogg'
        file_paths_ds = tf.data.Dataset.list_files(self.file_paths, shuffle=False)
        labels_ds = file_paths_ds.map(self.get_labels_from_path, num_parallel_calls=AUTOTUNE)
        ds = labels_ds.map(self.load_audio, num_parallel_calls=AUTOTUNE)
        return ds

if __name__ == '__main__':
    print("Testing SimpleDataset")
    print('-'*100)
    
    sd = SimpleDataset('../../input/birdclef-2021/train_short_audio/*/*.ogg')

    ds = sd.get_dataset()

    for row in ds:
        print(row)
        break
