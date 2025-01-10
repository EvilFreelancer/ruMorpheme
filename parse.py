import os
import requests
import bs4
import json
from random import choice
from tqdm import tqdm


def get_random_proxy() -> dict:
    url = "https://proxylist.geonode.com/api/proxy-list?limit=10&page=1&sort_by=lastChecked&sort_type=desc"
    response = requests.get(url)
    data = json.loads(response.text)
    proxies = []
    for item in data["data"]:
        protocol = item["protocols"][0]
        ip = item['ip']
        port = str(item['port'])
        proxies.append({protocol: f"{protocol}://{ip}:{port}"})

    proxy = choice(proxies)
    print(f"Random proxy: {proxy}")
    return proxy


CURRENT_PROXY = get_random_proxy()


def parse_span(span) -> list[dict]:
    """
    Рекурсивно обходит <span> и его дочерние <span>-элементы,
    извлекая корни, суффиксы, окончания и т.д.
    Возвращает список морфем вида [{"text": ..., "type": ...}, ...].

    Логику определения 'type' реализуем на основе классов:
      - root    -> ROOT
      - suffix  -> SUFF
      - prefix  -> PREF
      - ending  -> END  (если ending + nulled, то text = '0' или '')
      - и т.д.
    Сам класс "based" (основание) мы в конечный список **не добавляем**,
    но заходим вглубь и парсим дочерние элементы.
    """
    result = []
    classes = span.get("class", [])  # список классов у этого <span>
    text = span.get_text(strip=True)

    # Определяем тип по классам:
    if "root" in classes:
        result.append({"text": text, "type": "ROOT"})
    elif "suffix" in classes:
        result.append({"text": text, "type": "SUFF"})
    elif "prefix" in classes:
        result.append({"text": text, "type": "PREF"})
    elif "ending" in classes:
        # проверяем, есть ли "nulled" (нулевое окончание)
        if "nulled" in classes:
            result.append({"text": "", "type": "END"})
        else:
            result.append({"text": text, "type": "END"})
    elif "based" in classes:
        # "based" — это основание, но оно может содержать <span class="root"> и т.д.
        # Само "based" не добавляем как морфему, а парсим дочерние <span>.
        pass

    # Теперь обходим всех детей данного <span>, чтобы вытащить вложенные морфемы.
    for child in span.find_all("span", recursive=False):
        # recursive=False — берём только прямых детей.
        # Но если структура более глубокая, и там вложены еще <span>ы,
        # возможно, стоит убрать recursive=False.
        result.extend(parse_span(child))

    return result


def parse_morphemes_from_html(html: str) -> list[dict]:
    """
    Парсит HTML, находит первый div.morpheme,
    и извлекает морфемы (root, suffix, ending, prefix и т.д.)
    только до первого <br> (если <br> встретится).

    Возвращает список морфем [{"text": ..., "type": ...}, ...]
    в порядке появления в HTML.
    """
    soup = bs4.BeautifulSoup(html, "html.parser")

    # Ищем первый <div class="morpheme">
    div_morpheme = soup.find("div", class_="morpheme")
    if not div_morpheme:
        return []

    result = []

    # Перебираем всех "прямых" детей div'а (теги или текст),
    # но как только встретим <br>, прервёмся.
    for child in div_morpheme.children:
        # Если это тег <br>, то останавливаемся.
        if child.name == "br":
            break

        # Нас интересуют теги <span>, так как в них и лежат морфемы
        if child.name == "span":
            # parse_span рекурсивно достанет вложенные root/suffix/prefix...
            result.extend(parse_span(child))
        # Прочие текстовые узлы/комментарии игнорируем.

    return result


def get_morphemes(word: str) -> list[dict] | None:
    """
    Запрашиваем страницу для слова, парсим HTML и извлекаем морфемы до первого <br>.
    Если ничего не нашли, возвращаем None.
    """
    global CURRENT_PROXY

    url = f"https://morphemeonline.ru/{word[0].upper()}/{word}"
    max_retries = 3
    attempts = 0

    while attempts < max_retries:
        current_proxy = CURRENT_PROXY or get_random_proxy()

        try:
            r = requests.get(url, proxies=current_proxy, timeout=10)
            r.raise_for_status()

            # Новый парсер: берём HTML респонса и вытаскиваем морфемы
            morphemes = parse_morphemes_from_html(r.text)

            CURRENT_PROXY = current_proxy
            return morphemes if morphemes else None

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # print(f"Слово '{word}' не найдено (404). Пропускаем.")
                return None
            else:
                print(f"Ошибка HTTP при запросе слова '{word}': {e}. Пробуем ещё раз...")
                CURRENT_PROXY = None
                attempts += 1

        except requests.RequestException as e:
            print(f"Сетевая ошибка при запросе слова '{word}': {e}. Пробуем ещё раз...")
            CURRENT_PROXY = None
            attempts += 1

    print(f"Слово '{word}' пропущено после {max_retries} неудачных попыток.")
    return None


def main(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f]

    existing_words = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "word" in record:
                        existing_words.add(record["word"])
                except json.JSONDecodeError:
                    pass

    parsed_count = len(existing_words)
    total_count = len(words)

    with open(output_file, "a", encoding="utf-8") as f_out:
        with tqdm(total=total_count, initial=parsed_count, desc="Обработка слов", unit="слово") as pbar:
            for word in words:
                if word not in existing_words:
                    morphemes = get_morphemes(word)
                    if morphemes:
                        record = {"word": word, "morphemes": morphemes}
                    else:
                        record = {"word": word, "morphemes": None}

                    json.dump(record, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    existing_words.add(word)
                    pbar.update(1)


if __name__ == "__main__":
    # @see https://huggingface.co/datasets/Egor-AI/Russian-Words
    input_file = r"./russian-mnemonic-words.txt"
    output_file = r"./russian-mnemonic-words-morphemes.jsonl"

    main(input_file, output_file)
    print(f"Датасет сохранен в {output_file}.")
