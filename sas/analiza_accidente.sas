/* =======================================================================
PROIECT PACHETE SOFTWARE - ANALIZA ACCIDENTE RUTIERE SUA (PARTEA SAS)
========================================================================== */

/* Setăm SAS să convertească corect parantezele din CSV (ex: % sau C) în underscore _ */
OPTIONS VALIDVARNAME=V7;

/* 1. CREAREA ȘI FOLOSIREA DE FORMATE DEFINITE DE UTILIZATOR */
PROC FORMAT;
    VALUE sev_fmt
    1 = '1 - Minor'
    2 = '2 - Moderat'
    3 = '3 - Grav'
    4 = '4 - Critic';
RUN;

/* 2. CREAREA UNUI SET DE DATE SAS DIN FIȘIERE EXTERNE */
PROC IMPORT DATAFILE="/home/u64463762/Proiect/sample_procesat.csv"
    OUT=work.accidente_brut
    DBMS=CSV
    REPLACE;
    GUESSINGROWS=MAX;
RUN;

/* 3. SETĂRI ȘI PRELUCRARE - FUNCȚII SAS, CONDITII, ITERAȚII ȘI ARRAY */
DATA work.accidente_procesat;
    SET work.accidente_brut;

    /* 4. CREAREA DE SUBSETURI DE DATE (Filtrare) */
    WHERE Severity IS NOT MISSING;

    /* 5. UTILIZAREA DE FUNCȚII SAS */
    /* Extragem Anul si Luna. Start_Time este adus automat ca format DateTime de catre PROC IMPORT */
    Anul = YEAR(DATEPART(Start_Time));
    Luna = MONTH(DATEPART(Start_Time));

    /* 3b. PROCESARE CONDIȚIONALĂ (IF-THEN-ELSE) */
    IF Visibility_mi_ < 1 THEN Categorie_Vizibilitate = 'Slaba';
    ELSE IF Visibility_mi_ <= 5 THEN Categorie_Vizibilitate = 'Medie';
    ELSE Categorie_Vizibilitate = 'Buna';

    /* 7. UTILIZAREA DE MASIVE (ARRAY) + PROCESARE ITERATIVĂ (DO)*/
    /* Solutie: Rotunjim valorile valide la o zecimala pentru a respecta cerinta iteratiei,
       fara a polua datele cu -99. Daca e lipsa (.), ramane lipsa pentru a nu strica PROC MEANS. */
    ARRAY meteo_vars[*] Temperature_C_ Humidity___ Visibility_mi_ Wind_Speed_mph_;
    DO i = 1 TO DIM(meteo_vars);
        IF meteo_vars[i] ^= . THEN meteo_vars[i] = ROUND(meteo_vars[i], 0.1);
    END;

    DROP i;

    /* Aplicarea formatului user-defined */
    FORMAT Severity sev_fmt.;
RUN;

/* 6. COMBINAREA SETURILOR DE DATE (SQL și MERGE) */
/* a) Agregare via PROC SQL */
PROC SQL;
    CREATE TABLE work.metrici_stat AS
    SELECT
        State,
        COUNT(*) AS Numar_Accidente_Stat,
        MEAN(Severity) AS Severitate_Medie_Stat
    FROM work.accidente_procesat
    GROUP BY State;
QUIT;

/* b) Sortarea datelor pentru MERGE */
PROC SORT DATA=work.accidente_procesat; BY State; RUN;
PROC SORT DATA=work.metrici_stat; BY State; RUN;

/* c) Data Step MERGE */
DATA work.accidente_consolidat;
    MERGE work.accidente_procesat(IN=a) work.metrici_stat(IN=b);
    BY State;
    IF a AND b;
RUN;

/* 9. FOLOSIREA DE PROCEDURI STATISTICE */
PROC MEANS DATA=work.accidente_consolidat N MEAN MIN MAX STD;
    CLASS Severity;
    VAR Temperature_C_ Humidity___ Wind_Speed_mph_;
    TITLE 'Statistici Descriptive - Variabile Meteo pe Categorii de Severitate';
RUN;

PROC FREQ DATA=work.accidente_consolidat;
    TABLES Anul Sunrise_Sunset;
    TABLES Severity * Categorie_Vizibilitate / CHISQ;
    TITLE 'Distribuția Accidentelor și Analiza Bidimensională';
RUN;

/* 8. UTILIZAREA DE PROCEDURI PENTRU RAPORTARE */
PROC SORT DATA=work.metrici_stat OUT=work.metrici_stat_top;
    BY DESCENDING Numar_Accidente_Stat;
RUN;

PROC REPORT DATA=work.metrici_stat_top(OBS=15) NOWD HEADLINE HEADSKIP;
    COLUMN State Numar_Accidente_Stat Severitate_Medie_Stat;
    DEFINE State / ORDER 'Stat din SUA' WIDTH=15;
    DEFINE Numar_Accidente_Stat / DISPLAY 'Număr Accidente';
    DEFINE Severitate_Medie_Stat / DISPLAY 'Severitate Medie' FORMAT=8.2;
    TITLE 'Top 15 State cu cele mai multe accidente';
RUN;

/* 10. GENERAREA DE GRAFICE */
ODS GRAPHICS ON;
PROC SGPLOT DATA=work.accidente_procesat;
    VBAR Anul / GROUP=Severity GROUPDISPLAY=STACK;
    XAXIS LABEL='Anul de raportare';
    YAXIS LABEL='Număr total accidente';
    TITLE 'Distributia anuală a accidentelor grupate după severitate';
RUN;
ODS GRAPHICS OFF;
TITLE;

