def classify_emotional_state(hrv: dict) -> tuple[int, str]:
    """
    Классификация психо-эмоционального состояния по HRV-метрикам.

    Используем:
      - SDNN  (общая вариабельность)
      - RMSSD (кратковременная вариабельность)
      - pNN50 (доля интервалов >50мс)
      - CSI   (Cardiac Sympathetic Index)
      - CVI   (Cardiac Vagal Index)
    """
    sdnn   = hrv.get('sdnn', 0)
    rmssd  = hrv.get('rmssd', 0)
    pnn50  = hrv.get('pnn50', 0)
    csi    = hrv.get('csi', 0)
    cvi    = hrv.get('cvi', 0)

    # Правила: (index_emo, условие, текст)
    rules = [
        (1, lambda: sdnn >= 200 and rmssd >= 200 and pnn50 >= 90 and csi < 1.0 and cvi < 5,
         "участник максимально расслаблен, полное отсутствие тревоги"),
        (2, lambda: sdnn >= 100 and rmssd >= 100 and pnn50 >= 75 and csi < 1.1,
         "участник расслаблен, уровень стресса низкий"),
        (3, lambda: sdnn >=  80 and rmssd >=  50 and pnn50 >= 50,
         "участник в лёгком фокусе, умеренная концентрация"),
        (4, lambda: sdnn >=  60 and rmssd >=  30,
         "активная бдительность, полная вовлечённость"),
        (5, lambda: sdnn >=  40 and rmssd >=  20,
         "умеренное напряжение, возможная усталость"),
        (6, lambda: csi >= 1.1 and cvi >= 5,
         "стресс: высокая умственная или эмоциональная нагрузка"),
        (7, lambda: csi >= 1.2,
         "тревожность: повышенная нервозность, гипервозбуждение"),
        (8, lambda: csi >= 1.5,
         "предвыгорание: снижение продуктивности, раздражительность"),
        (9, lambda: csi >= 2.0,
         "выгорание: хроническое истощение, полное отсутствие ресурса"),
    ]

    for index_emo, cond, text in rules:
        if cond():
            return index_emo, text

    return 0, "состояние не определено"
