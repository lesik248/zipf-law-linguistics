import sqlite3
import matplotlib.pyplot as plt
import numpy as np

LANGS = ["ru", "en", "fr"]

def zipfs_law(db_cursor, lang):
    db_cursor.execute("SELECT freq, row_number() over(order by freq desc) as rank FROM dict where lang = ? limit 1000", (lang,))
    dict_data = db_cursor.fetchall()

    freqs = []
    ranks = []
    for (freq, rank) in dict_data:
        freqs.append(freq)
        ranks.append(rank)

    return (freqs, ranks)

def empiric_zipfs_law(db_cursor, lang):
    db_cursor.execute("SELECT freq, dict.word " \
    "FROM dict " \
    "inner join service_words s on s.word = dict.word " \
    "where lang = ?" \
    "order by freq desc limit 100", (lang,))
    dict_data = db_cursor.fetchall()

    freqs = []
    words_lengths = []
    for (freq, word) in dict_data:
        freqs.append(freq)
        words_lengths.append(len(word))

    return (np.array(freqs), np.array(words_lengths))

from collections import defaultdict
import numpy as np

def zhuyans_coefficient(db_cursor, lang):
    db_cursor.execute("SELECT MAX(part) FROM dict_parts WHERE lang = ?", (lang,))
    parts_count = db_cursor.fetchone()[0]

    db_cursor.execute("""
        SELECT part, word, freq,
               ROW_NUMBER() OVER (PARTITION BY part ORDER BY freq DESC) AS rank
        FROM dict_parts
        WHERE lang = ?
    """, (lang,))
    rows = db_cursor.fetchall()

    data = defaultdict(lambda: {'freq': [], 'rank': []})
    for part, word, freq, rank in rows:
        data[word]['freq'].append(freq)
        data[word]['rank'].append(rank)

    results = []
    for word, values in data.items():
        frequencies = values['freq']
        ranks = values['rank']

        avg_freq = np.mean(frequencies)
        std_dev_freq = np.std(frequencies)
        avg_rank = np.mean(ranks)

        if parts_count <= 1 or avg_freq == 0:
            coefficient = 0
        else:
            coefficient = 100 * (1 - std_dev_freq / (avg_freq * np.sqrt(parts_count - 1)))

        results.append((word, avg_freq, avg_rank, coefficient, lang))

    db_cursor.executemany("""
        INSERT INTO zhuyns_coeff (word, avg_freq, avg_rank, D, lang)
        VALUES (?, ?, ?, ?, ?)
    """, results)

    db_cursor.connection.commit()
    print('processed language')



def plot_graph(langs_data):
    plt.figure(figsize=(8,6))
    colors = ['red', 'green', 'blue']
    for i in range(len(langs_data)):
        (freqs, ranks) = langs_data[i]
        plt.plot(freqs, ranks, marker='o', color = colors[i], ms = 0.07, label = LANGS[i])

    plt.xlabel('Ранг, r')
    plt.ylabel('Частота, f')
    plt.title("Закон Ципфа (f*r=c)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_empiric_zipf(word_lengths, freqs):
    plt.figure(figsize=(8,6))
    plt.scatter(word_lengths, freqs, alpha=0.6)

    plt.xscale('log')
    plt.yscale('log')

    coeffs = np.polyfit(np.log(word_lengths), np.log(freqs), 1)
    k, b = coeffs
    plt.plot(word_lengths, np.exp(b) * word_lengths ** k, color='red', label=f'k={k:.2f}')

    plt.xlabel('Длина слова')
    plt.ylabel('Частота (log scale)')
    plt.title('Эмпирический закон Ципфа для служебных частей речи')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    connection = sqlite3.connect('./data/freqdict.db')
    cursor = connection.cursor()

    zipfs_law_data = []
    empiric_zipfs_law_data = []
    for lang in LANGS:
        (freqs, ranks) = zipfs_law(cursor, lang)
        zipfs_law_data.append((freqs, ranks))
        (serv_words_freqs, words_lengths) = empiric_zipfs_law(cursor, lang)
        empiric_zipfs_law_data.append((serv_words_freqs, words_lengths))
        # zhuyans_coefficient(cursor, lang)

    plot_graph(zipfs_law_data)
    plot_graph(empiric_zipfs_law_data)

    connection.close()