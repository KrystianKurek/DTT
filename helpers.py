import tqdm
from wordcloud import WordCloud
from numba import njit, prange
from numba.typed import Dict
from numba.types import int64, int32
import networkx as nx

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import compute_class_weight


def plot_categories(column, ax=None):
    values_counter = Counter(column.dropna())
    column_values = list(values_counter.keys())
    values_count = list(values_counter.values())

    if ax is None:
        fig, ax = plt.subplots()

    if len(column_values) < 10:
        sns.barplot(x=column_values, y=values_count, ax=ax)
        ax.set_xlabel('Values')
        ax.set_ylabel('Count')
        ax.set_title(column.name)
        for i, v in enumerate(values_count):
            ax.text(i, v, str(v), ha='center', va='bottom')
    else:
        wordcloud = WordCloud(include_numbers=True, width=600, height=400, background_color=None, mode="RGBA") \
            .generate_from_frequencies(values_counter)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')


def print_number_of_nans(df):
    print("Rows with nans in columns: ")
    for col in df.columns:
        nan_num = df[col].isna().sum()
        if nan_num:
            print(col, nan_num)


def print_number_of_distinct_values(df):
    print("Number of distinct values in columns: ")
    for col in df.columns:
        num_distinct = len(df[col].drop_duplicates())
        if num_distinct:
            print(col, num_distinct)


@njit
def combinations(values):
    """This function works similarly to itertools.combinations for pairs"""
    pairs = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((values[i], values[j]))
    return pairs


@njit(parallel=True)
def build_graph_inner(df_raw):
    unique_values = np.unique(df_raw.flatten())
    value_to_node_id = Dict.empty(
        key_type=int32,
        value_type=int32,
    )
    for i, val in enumerate(unique_values):
        value_to_node_id[val] = i

    n_nodes = len(unique_values)
    am = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for i in prange(df_raw.shape[0]):
        row_values = df_raw[i]
        possible_pairs = combinations(row_values)
        for pair in possible_pairs:
            val_x, val_y = pair
            node_x, node_y = value_to_node_id[val_x], value_to_node_id[val_y]
            am[node_x, node_y] += 1
            am[node_y, node_x] += 1
    return am, value_to_node_id


def build_graph(df):
    df = df.copy()
    all_values_to_int = {}
    for i, col in enumerate(df.columns):
        values_to_int = {val: i * 100000 + val_index for val_index, val in enumerate(np.unique(df[col]))}
        df[col] = df[col].map(values_to_int)
        all_values_to_int.update(values_to_int)
    am, mapping = build_graph_inner(df.values)
    mapping = dict(mapping)
    all_values_to_node_id = {value: mapping[int_] for value, int_ in all_values_to_int.items()}
    return nx.from_numpy_array(am), all_values_to_node_id

@tf.function
def one_hot(name, names_list, depth):
    idx = tf.where(tf.equal(names_list, name))
    if tf.size(idx) == 0:
        return tf.zeros([1, depth])
    else:
        idx = idx[0]
        return tf.one_hot([idx], depth=depth)


def one_hot_encode_batch(batch, labels, categories, class_weights):
    batch_encoded = []
    for col_idx, (col, cats) in enumerate(categories.items()):
        one_hot_encoded = one_hot(batch[col_idx], cats, len(cats))
        one_hot_encoded = tf.reshape(one_hot_encoded, (1, -1))
        batch_encoded.append(one_hot_encoded)
    batch_encoded = tf.concat(batch_encoded, axis=1)
    return tf.reshape(batch_encoded, (-1,)), labels, class_weights.lookup(tf.cast(labels, tf.int32))


def create_tf_dataset(df, y, batch_size, categories, class_weights):
    dataset = tf.data.Dataset.from_tensor_slices((df.values, y.values)) \
        .shuffle(len(df)) \
        .map(lambda x, y: one_hot_encode_batch(x, y, categories, class_weights)) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .repeat(99999) \
        .apply(tf.data.experimental.prefetch_to_device("/gpu:0"))

    return dataset


def return_compiled_network(input_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history):
    fig, axs = plt.subplots(nrows=2, figsize=(18, 6))

    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def train_and_save_model(X, y, nn_embedding_file_model, batch_size=256):
    categories = {col: X[col].drop_duplicates().tolist() for col in X.columns}

    n = len(sum(categories.values(), []))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([0, 1]),
            values=tf.constant(list(class_weights))
        ),
        default_value=tf.constant(-1.0, dtype=tf.float64)
    )

    train_generator = create_tf_dataset(X_train, y_train, batch_size, categories, class_weights)
    val_generator = create_tf_dataset(X_val, y_val, batch_size, categories, class_weights)

    model = return_compiled_network(n)
    # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=2),
    #             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001)]
    history = model.fit(x=train_generator, batch_size=batch_size, epochs=3, validation_data=val_generator,
                        use_multiprocessing=True, workers=4,
                        validation_steps=len(X_val) // batch_size + 1,
                        steps_per_epoch=len(X_train) // batch_size + 1)

    plot_history(history)

    layer_output = model.layers[-2].output
    emb_model = tf.keras.Model(inputs=model.input, outputs=layer_output)

    emb_model.save(nn_embedding_file_model)


def inference_one_hot_encode_batch(batch, categories):
    batch_encoded = []
    for col_idx, (col, cats) in enumerate(categories.items()):
        one_hot_encoded = one_hot(batch[col_idx], cats, len(cats))
        one_hot_encoded = tf.reshape(one_hot_encoded, (1, -1))
        batch_encoded.append(one_hot_encoded)
    batch_encoded = tf.concat(batch_encoded, axis=1)
    return tf.reshape(batch_encoded, (-1,))


def create_inference_tf_dataset(df, batch_size, categories):
    dataset = tf.data.Dataset.from_tensor_slices(df.values) \
        .map(lambda x: inference_one_hot_encode_batch(x, categories)) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_nn_embeddings(df, model, categories=None):
    if categories is None:
        categories = {col: df[col].drop_duplicates().tolist() for col in df.columns}

    dataset = create_inference_tf_dataset(df, 64, categories)
    all_embeddings = []
    for batch in tqdm.tqdm(dataset, total=len(df) // 64 + 1):
        with tf.device("/cpu:0"):
            batch_embeddings = model(batch)
            all_embeddings.append(batch_embeddings)

    return np.concatenate(all_embeddings, axis=0)


def get_graph_embeddings(df, graph_model, mapping, fillna_graph_methods):
    max_value = max(mapping.values())
    node_id_to_node_emb = {node_id: list(graph_model.wv[(node_id,)].flatten()) for node_id in range(max_value + 1)}

    for col in df.columns:
        df[col] = df[col].map(mapping).map(node_id_to_node_emb)
        df[col] = df[col].apply(lambda x: fillna_graph_methods if isinstance(x, float) else x)  # can't fillna with lists :(

    return np.array(df.apply(lambda row: sum(row, []), axis=1).tolist())
