def translate_hebrew(hebrew_text):
    translations = {
        'לא מסכים': 'Disagree',
        'מסכים': 'Agree',
        'מאוד': 'Very',
        'לפעמים': 'Sometimes',
        'לעיתים נדירות': 'Rarely',
        'בכלל לא': 'Not at all',
        'כן בהחלט': 'Yes definitely',
        'כן בערך': 'Yes roughly',
        'לא ממש': 'Not really',
        'דרוך': 'Tense',
        'מעט': 'A little',
        'במקצת': 'Somewhat',
        'במידה בינונית': 'Moderately',
        'ככלל לא': 'Not at all',
        'בדרך כלל': 'Usually',
        'כל הזמן': 'All the time',
        'לעתים רחוקות': 'Rarely',
        'במידה מסוימת': 'To a certain extent',
        'במידה מתונה': 'To a moderate extent',
        'מא': 'Very',
        'אני לא מסכים': 'I disagree',
        'בהחלט לא מסכים': 'Strongly disagree',
        'מסכים בהחלט': 'Strongly agree',
        'בפנים עם נוף נעים': 'Inside with a pleasant view',
        'בתוך הבית ללא נוף נעים': 'Inside the house without a pleasant view',
        'בחוץ עם נוף נעים': 'Outside with a pleasant view',
        'בחוץ ללא נוף נעים': 'Outside without a pleasant view',
        'כן': 'Yes',
        'לא': 'No',
        'מסכים בהחלט': 'Strongly agree',
        'ערני': 'Alert',
        'מותש': 'Exhausted',
        'עצוב': 'Sad',
        'מלא התרגשות': 'Full of excitement',
        'מרגיש סיפוק': 'Feeling satisfied',
        'דרוך/ה': 'Tense',
        'רגוע/ה': 'Calm',
        'מוטרד/ת': 'Troubled',
        'שביעות רצון': 'Satisfaction',
        'מודאג/ת': 'Worried',
    }
    
    return translations.get(hebrew_text, hebrew_text)
