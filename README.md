Вот готовый текст для `README.md` вашего репозитория — с кратким описанием, установкой, запуском и сервисом. Можете просто скопировать как есть.

---

# MusicGen → AzuraCast (через API) — «флот» генерируемых треков

Скрипт непрерывно генерирует музыку с помощью **Facebook Audiocraft / MusicGen**, кодирует в **MP3** (через `ffmpeg`), проставляет **ID3**-теги, загружает треки в медиатеку **AzuraCast** через API, сразу добавляет их в указанный **плейлист** и поддерживает «флот» из N треков (например, 40–50) с **ежедневным освежением** части треков (например, 10 в сутки).

* Генерация по `--prompt` (модели `facebook/musicgen-*`)
* MP3 кодирование (44.1 kHz Stereo, `libmp3lame`, настраиваемый битрейт)
* ID3 (фреймы `TIT2`/`TPE1`/`TALB`/`COMM`)
* Загрузка через AzuraCast API + привязка к плейлисту
* «Флот» (минимум/максимум), ежедневное обновление в заданное время
* Состояние — `out/.state.json` (медиатеки, ID, даты загрузки и т.д.)

---

## Требования

* Linux (тестировалось на Ubuntu 22.04/24.04)
* **Python 3.10+**
* **ffmpeg** в системе
* **AzuraCast** с включённым API, API key, Station ID, целевой плейлист
* **CUDA (опционально)** — если хотите ускорить генерацию на GPU
  Совместимая сборка PyTorch под вашу версию драйвера/NVIDIA CUDA Runtime

### Python зависимости

Скрипт использует:

* `audiocraft==1.3.0`
* `transformers>=4.38`
* `requests`
* `mutagen`
* `torch / torchaudio / torchvision` — под ваш CPU/CUDA стек
* `numpy<2` (надёжно для Audiocraft 1.3.0)

> ⚠️ Для CUDA ставьте **torch/torchaudio/torchvision** из индекса PyTorch под вашу CUDA (см. ниже).

---

## Установка

```bash
git clone https://github.com/reinethernal/musicgen_uploader_api.git musicgen
cd musicgen

# 1) FFmpeg (если ещё не установлен)
sudo apt-get update && sudo apt-get install -y ffmpeg

# 2) Python venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel

# 3) Базовые зависимости
cat > requirements.txt <<'REQ'
numpy<2
audiocraft==1.3.0
transformers>=4.38
requests
mutagen
REQ
pip install -r requirements.txt
```

### Вариант A: CPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Вариант B: CUDA

1. Узнайте, какую линейку CUDA поддерживает ваш драйвер (например, CUDA 12.1/12.4/12.6).
2. Установите соответствующие колёса PyTorch. Пример для **CUDA 12.1**:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121
```

> Если у вас другая версия (например, cu124, cu126), замените индекс/версии согласно документации PyTorch.

Проверка:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
PY
```

---

## Настройка AzuraCast

Вам понадобятся:

* **Base URL** вашей установки (например, `https://radio.example.com`)
* **API Key** (создаётся в «My Account → API Keys» или в админке)
* **Station ID** (числовой ID станции)
* Имя **плейлиста**, куда класть новые треки (например, `incoming`)
* Путь **в медиатеке** станции (обычно `/incoming`)

> Убедитесь, что у API-ключа достаточно прав для загрузки файлов и управления плейлистами.

---

## Запуск (пример)

```bash
source venv/bin/activate

python musicgen_uploader_api.py \
  --prompt "Cinematic ambient music with atmospheric synth pads and slow-evolving textures creates a serene, immersive backdrop for deep focus and productivity." \
  --model facebook/musicgen-small \
  --duration 120 \
  --bitrate 192k \
  --outdir ./out --local-keep 60 \
  --fleet-min 40 --fleet-max 50 \
  --daily-refresh 10 --daily-at 03:30 \
  --tick-sec 15 \
  --ac-base-url https://radio.example.com \
  --ac-api-key 'USERID:APIKEY' \
  --ac-station-id 2 \
  --ac-path /incoming \
  --ac-playlist incoming \
  --cuda --gpu 0
```

* Скрипт заполнит «флот» до **минимума** (`--fleet-min`), но не превысит **максимум** (`--fleet-max`).
* Раз в сутки в указанное время (`--daily-at`) удалит **N** старых (`--daily-refresh`) и дозальёт новые.
* Локально хранит до `--local-keep` файлов (остальные удаляет).

Файлы состояния: `out/.state.json`
Логи — в stdout (удобно запускать в `screen`/`tmux` или как сервис).

---

## Параметры (ключевые)

* `--prompt` — текстовый запрос к MusicGen (обязательно)
* `--model` — модель (`facebook/musicgen-small|medium|large|melody`)
* `--duration` — длительность трека в секундах (например, 120)
* `--bitrate` — битрейт MP3 (`96k|128k|192k|...`)
* `--outdir` — локальный каталог для сохранения
* `--local-keep` — сколько локальных файлов держать
* `--fleet-min / --fleet-max` — целевой коридор размера «флота»
* `--daily-refresh` — сколько удалять/добавлять раз в сутки
* `--daily-at` — время ежедневного обновления (локальное, `HH:MM`)
* `--tick-sec` — периодичность цикла обслуживания
* `--ac-base-url` — базовый URL AzuraCast
* `--ac-api-key` — API-ключ (`userId:apiKey` или просто ключ — как настроено)
* `--ac-station-id` — ID станции
* `--ac-path` — путь в медиатеке, например `/incoming`
* `--ac-playlist` — имя плейлиста для привязки
* `--cuda` `--gpu N` — включить CUDA и выбрать GPU по индексу

---

## Запуск как systemd-сервис (опционально)

1. **Env-файл** `/etc/musicgen/uploader.env`:

```bash
PYTHON=/root/musicgen-azuracast/venv/bin/python
WORKDIR=/root/musicgen-azuracast

PROMPT='Cinematic ambient music with atmospheric synth pads and slow-evolving textures creates a serene, immersive backdrop for deep focus and productivity.'
MODEL='facebook/musicgen-small'
DURATION=120
BITRATE=192k
OUTDIR=/root/musicgen-azuracast/out
LOCAL_KEEP=60
FLEET_MIN=40
FLEET_MAX=50
DAILY_REFRESH=10
DAILY_AT=03:30
TICK_SEC=15

AC_BASE_URL='https://radio.example.com'
AC_API_KEY='USERID:APIKEY'
AC_STATION_ID=2
AC_PATH='/incoming'
AC_PLAYLIST='incoming'

USE_CUDA=1
GPU=0
```

2. **Wrapper** `/usr/local/bin/musicgen-uploader`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /root/musicgen/uploader.env
cd "$WORKDIR"

ARGS=(
  --prompt "$PROMPT"
  --model "$MODEL"
  --duration "$DURATION"
  --bitrate "$BITRATE"
  --outdir "$OUTDIR" --local-keep "$LOCAL_KEEP"
  --fleet-min "$FLEET_MIN" --fleet-max "$FLEET_MAX"
  --daily-refresh "$DAILY_REFRESH" --daily-at "$DAILY_AT"
  --tick-sec "$TICK_SEC"
  --ac-base-url "$AC_BASE_URL"
  --ac-api-key "$AC_API_KEY"
  --ac-station-id "$AC_STATION_ID"
  --ac-path "$AC_PATH"
  --ac-playlist "$AC_PLAYLIST"
)

if [[ "${USE_CUDA:-0}" == "1" ]]; then
  ARGS+=( --cuda --gpu "$GPU" )
fi

exec "$PYTHON" musicgen_uploader_api.py "${ARGS[@]}"
```

> ВАЖНО: файл должен быть в **LF** (Unix)-формате. Если видите ошибку `‘bash\r’: No such file or directory`, конвертируйте:
>
> ```bash
> sudo sed -i 's/\r$//' /usr/local/bin/musicgen-uploader /root/musicgen/uploader.env
> ```

3. **Unit-файл** `/etc/systemd/system/musicgen-uploader.service`:

```ini
[Unit]
Description=MusicGen -> AzuraCast fleet uploader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5
ExecStart=/usr/local/bin/musicgen-uploader
WorkingDirectory=/root/musicgen-azuracast
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Запуск:

```bash
sudo chmod 755 /usr/local/bin/musicgen-uploader
sudo systemctl daemon-reload
sudo systemctl enable --now musicgen-uploader
journalctl -u musicgen-uploader -f -n 200
```

---

## Как работает «флот»

* Скрипт следит, чтобы число треков было **между** `fleet-min` и `fleet-max`.
* Если меньше минимума — генерирует и дозаливает.
* Если больше максимума — удаляет самые старые (и локально, и в AzuraCast).
* Раз в сутки в `--daily-at` удаляет `--daily-refresh` **самых старых** и генерирует столько же новых.
* ID загруженных медиа и базовые метаданные хранятся в `out/.state.json`.

---

## Советы и отладка

* Если появился `405 Method Not Allowed` на `/files/sync` — этот endpoint не нужен: скрипт использует поддерживаемые `/files/upload`, `/files` и `/file/{id}`.
* Если файл не сразу виден — индексатор AzuraCast может отрабатывать с небольшой задержкой. Скрипт сам будет повторно искать `media_id`.
* Если видите предупреждения `numpy 2.x`/`A module compiled for NumPy 1.x…` — придерживайтесь `numpy<2` для `audiocraft==1.3.0`.
* CUDA в контейнерах (Proxmox LXC и т.п.) требует корректных девайсов: **/dev/nvidia0**, **/dev/nvidiactl**, **/dev/nvidia-uvm** (не только `-tools`).
  Проверьте `nvidia-smi`, доступ к устройствам и соответствие версии PyTorch вашей CUDA.
* Безопасность: не коммитьте **API-ключ** в репозиторий, храните его в env-файле/секретах.

---

## Лицензия

Добавьте сюда текст вашей лицензии (например, MIT). Если не указать лицензию, проект считается «все права защищены».

---

## Пример: быстрый старт (CPU)

```bash
git clone <ВАШ_GIT_URL> musicgen-azuracast && cd musicgen-azuracast
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python musicgen_uploader_api.py \
  --prompt "Cinematic ambient music with atmospheric synth pads and slow-evolving textures creates a serene, immersive backdrop for deep focus and productivity." \
  --model facebook/musicgen-small \
  --duration 120 --bitrate 192k \
  --outdir ./out --local-keep 60 \
  --fleet-min 40 --fleet-max 50 \
  --daily-refresh 10 --daily-at 03:30 \
  --tick-sec 15 \
  --ac-base-url https://radio.example.com \
  --ac-api-key 'USERID:APIKEY' \
  --ac-station-id 2 \
  --ac-path /incoming \
  --ac-playlist incoming
```

Готово. Если нужно — поднимайте как `systemd`-сервис (см. выше).
