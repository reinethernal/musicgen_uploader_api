# MusicGen → AzuraCast (fleet uploader)

Сервис генерирует треки при помощи **Facebook Audiocraft / MusicGen**, кодирует их в **MP3**, проставляет **ID3**, загружает в медиатеку **AzuraCast** через API и автоматически добавляет в указанный плейлист. Скрипт поддерживает «флот» треков (минимум/максимум) и ежедневное освежение части каталога.

## Структура

* `/root/musicgen/` — директория проекта

  * `musicgen_uploader_api.py` — основной генератор/загрузчик
  * `uploader.env` — конфиг (лежит в директории проекта)
  * `venv/` — виртуальное окружение Python
  * `out/` — локальные MP3 и файл состояния `.state.json`
* `/usr/local/bin/musicgen-uploader` — обёртка (bash), запускает Python-скрипт по значениям из `uploader.env`
* `systemd` unit: `musicgen-uploader.service`

## Требования

* Linux (Ubuntu 22.04/24.04 и др.)
* **Python 3.10+**
* **ffmpeg** установлен в системе
* **AzuraCast** с API-ключом и Station ID, существующий плейлист
* (опц.) **CUDA** + совместимая сборка PyTorch — если хотите генерировать на GPU

## Установка

1. FFmpeg + Python venv:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
python3 -m venv /root/musicgen/venv
source /root/musicgen/venv/bin/activate
pip install --upgrade pip wheel
```

2. Python-зависимости:

```bash
# базовые библиотеки
pip install 'numpy<2' audiocraft==1.3.0 'transformers>=4.38' requests mutagen
```

3. PyTorch (выберите один вариант):

* CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

* CUDA (пример для CUDA 12.1; подставьте свою линейку, если другая):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.1.0+cu121" "torchvision==0.16.0+cu121" "torchaudio==2.1.0+cu121"
```

Проверка:

```bash
python - <<'PY'
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY
```

## Конфигурация (`/root/musicgen/uploader.env`)

Создайте файл и заполните переменные окружения:

```bash
# Генерация
export MUSICGEN_PROMPT='Cinematic ambient music with atmospheric synth pads and slow-evolving textures creates a serene, immersive backdrop for deep focus and productivity.'
export MUSICGEN_MODEL='facebook/musicgen-small'     # можно medium/large/melody при достаточном VRAM
export MUSICGEN_DURATION=120                        # сек/трек
export MUSICGEN_BITRATE='192k'                      # mp3 битрейт

# Локальное хранилище
export MUSICGEN_OUTDIR='/root/musicgen/out'
export MUSICGEN_LOCAL_KEEP=60                       # сколько локальных MP3 держать

# «Флот» в AzuraCast
export MUSICGEN_FLEET_MIN=40
export MUSICGEN_FLEET_MAX=50
export MUSICGEN_DAILY_REFRESH=10                    # ежедневная замена
export MUSICGEN_DAILY_AT='03:30'                    # локальное время HH:MM
export MUSICGEN_TICK_SEC=15

# AzuraCast API
export MUSICGEN_AC_BASE_URL='https://radio.example.com'
export MUSICGEN_AC_API_KEY='USERID:APIKEY'          # как выдается вашей инсталляцией
export MUSICGEN_AC_STATION_ID=2
export MUSICGEN_AC_PATH='/incoming'
export MUSICGEN_AC_PLAYLIST='incoming'

# CUDA (0 — выключено; 1 — включено)
export MUSICGEN_CUDA=1
export MUSICGEN_GPU=0
```

> Рекомендуемые права на конфиг:
>
> ```bash
> sudo chown root:root /root/musicgen/uploader.env
> sudo chmod 600 /root/musicgen/uploader.env
> ```

## Обёртка `/usr/local/bin/musicgen-uploader`

Содержимое:

```bash
#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="/root/musicgen/uploader.env"
[[ -f "$ENV_FILE" ]] || { echo "Config $ENV_FILE not found"; exit 1; }
# shellcheck disable=SC1090
. "$ENV_FILE"

VENV_PY="/root/musicgen/venv/bin/python"
SCRIPT="/root/musicgen/musicgen_uploader_api.py"

EXTRA=()
if [[ "${MUSICGEN_CUDA:-0}" == "1" ]]; then
  EXTRA+=(--cuda --gpu "${MUSICGEN_GPU:-0}")
fi

install -d -m 755 "$(dirname "$MUSICGEN_OUTDIR")" "$MUSICGEN_OUTDIR"

exec "$VENV_PY" "$SCRIPT" \
  --prompt "$MUSICGEN_PROMPT" \
  --model "$MUSICGEN_MODEL" \
  --duration "$MUSICGEN_DURATION" \
  --bitrate "$MUSICGEN_BITRATE" \
  --outdir "$MUSICGEN_OUTDIR" --local-keep "$MUSICGEN_LOCAL_KEEP" \
  --fleet-min "$MUSICGEN_FLEET_MIN" --fleet-max "$MUSICGEN_FLEET_MAX" \
  --daily-refresh "$MUSICGEN_DAILY_REFRESH" --daily-at "$MUSICGEN_DAILY_AT" \
  --tick-sec "$MUSICGEN_TICK_SEC" \
  --ac-base-url "$MUSICGEN_AC_BASE_URL" \
  --ac-api-key "$MUSICGEN_AC_API_KEY" \
  --ac-station-id "$MUSICGEN_AC_STATION_ID" \
  --ac-path "$MUSICGEN_AC_PATH" \
  --ac-playlist "$MUSICGEN_AC_PLAYLIST" \
  "${EXTRA[@]}"
```

Установите права и убедитесь, что файл в LF-формате (не DOS-CRLF):

```bash
sudo tee /usr/local/bin/musicgen-uploader >/dev/null < musicgen-uploader
sudo sed -i 's/\r$//' /usr/local/bin/musicgen-uploader
sudo chmod 755 /usr/local/bin/musicgen-uploader
```

## Запуск вручную

```bash
/usr/local/bin/musicgen-uploader
```

Логи — в stdout. Локальные MP3 и состояние — в `/root/musicgen/out/`.

## Запуск как systemd-сервис

Создайте юнит `/etc/systemd/system/musicgen-uploader.service`:

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
WorkingDirectory=/root/musicgen
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Активация:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now musicgen-uploader
journalctl -u musicgen-uploader -f -n 200
```

> Если видите `/usr/bin/env: ‘bash\r’: No such file or directory`, конвертируйте CRLF → LF:
>
> ```bash
> sudo sed -i 's/\r$//' /usr/local/bin/musicgen-uploader /root/musicgen/uploader.env
> ```

## Как это работает

* Скрипт поддерживает «флот» из `MUSICGEN_FLEET_MIN..MUSICGEN_FLEET_MAX` треков в медиатеке станции.
* Раз в сутки в `MUSICGEN_DAILY_AT` удаляет `MUSICGEN_DAILY_REFRESH` самых старых треков и генерирует столько же новых.
* Привязка загруженных файлов к плейлисту выполняется через AzuraCast API.
* Локальная директория чистится, чтобы оставалось не более `MUSICGEN_LOCAL_KEEP` MP3.

## Подсказки и отладка

* **API-ключ** храните только в `uploader.env`, права 600.
* Если файл загрузился, но сразу не виден — у AzuraCast может быть небольшая задержка индексатора. Скрипт сам подберёт `media_id` позже.
* Ошибка `numpy 2.x`/`A module compiled for NumPy 1.x…` — используйте `numpy<2` с `audiocraft==1.3.0`.
* Для CUDA в контейнерах убедитесь, что проброшены `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, и версии PyTorch соответствуют вашей линейке CUDA.

Готово!
