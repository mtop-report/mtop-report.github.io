/* MTOP "Standards" tab: the reporting guidelines MTOP follows (TRIPOD+AI and its abstract checklist),
 * a PROBAST+AI self-assessment, the sample-size criteria (artifacts/public/reporting.json), the other
 * standards that apply now or later, and what is still to do. Checklist items are paraphrased; see the
 * original publications for the exact wording. */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A) return;
  const { h, tx, num, L } = A;
  const S = A.S;
  S.repFilter = S.repFilter || "all";
  const R = A.DATA.reporting || null;

  // where in MTOP an item is addressed: [view, label en, label cs, extra state]
  const W = {
    data: ["registry", "Registry → Data and preprocessing", "Registr → Data a předzpracování", { regView: "data" }],
    registry: ["registry", "Registry", "Registr", { regView: "outcomes" }],
    val: ["validation", "Validation", "Validace"],
    calc: ["calc", "Calculator", "Kalkulačka"],
    adapt: ["adaptive", "Adaptive", "Adaptace"],
    multi: ["validate", "Multicentre", "Multicentrum"],
    fed: ["federated", "Multicentre → federated", "Multicentrum → federované"],
    road: ["roadmap", "Roadmap", "Plán"],
    about: ["about", "About", "O projektu"],
    std: ["reporting", "This page", "Tato stránka"],
    repo: [null, "Code repository", "Repozitář kódu"],
    ms: [null, "Manuscript", "Rukopis"],
  };

  // TRIPOD+AI (Collins et al., BMJ 2024;385:e078378): 27 items, 52 sub-items. Paraphrased.
  // [item, section, what it asks (en, cs), status, where[], what to do (en, cs)]
  const TRIPOD = [
    ["1", "Title", ["Name the study as development and/or evaluation of a prediction model, with the target population and the outcome.", "Označit studii jako vývoj a/nebo hodnocení predikčního modelu, s cílovou populací a výsledkem."], "todo", ["ms"],
      ["For example: “Development and temporal validation of MTOP, a staged model of 90-day outcome after mechanical thrombectomy”.", "Například: „Vývoj a časová validace MTOP, postupného modelu výsledku 90 dní po mechanické trombektomii“."]],
    ["2", "Abstract", ["Follow TRIPOD+AI for Abstracts.", "Postupovat podle TRIPOD+AI pro abstrakty."], "todo", ["ms", "std"], ["Use the 13-item list below.", "Použít 13 položek níže."]],
    ["3a", "Background", ["Healthcare context and rationale, with reference to existing models.", "Klinický kontext a zdůvodnění, s odkazem na existující modely."], "done", ["val"],
      ["Published scores (THRIVE, SPAN-100, HIAT2, PRE, MR PREDICTS@24H) are benchmarked on the same patients; cite them.", "Publikovaná skóre (THRIVE, SPAN-100, HIAT2, PRE, MR PREDICTS@24H) jsou srovnána na stejných pacientech; citovat je."]],
    ["3b", "Background", ["Target population, intended purpose and moment of use in the care pathway, and the intended users.", "Cílová populace, účel a okamžik použití v péči a zamýšlení uživatelé."], "partly", ["calc", "about"],
      ["State the users (stroke neurologists, interventional radiologists) and the purpose: prognostic counselling at four time points, not treatment selection.", "Uvést uživatele (neurologové, intervenční radiologové) a účel: prognostická informace ve čtyřech okamžicích, nikoli volba léčby."]],
    ["3c", "Background", ["Known health inequalities between sociodemographic groups.", "Známé nerovnosti ve zdraví mezi sociodemografickými skupinami."], "todo", ["ms"],
      ["Summarise what is known about sex and age differences in outcome after thrombectomy.", "Shrnout známé rozdíly výsledku po trombektomii podle pohlaví a věku."]],
    ["4", "Objectives", ["Objectives, including whether the study develops, evaluates or both.", "Cíle, včetně toho, zda jde o vývoj, hodnocení nebo obojí."], "done", ["about"],
      ["Development, temporal validation (primary and sensitivity split) and comparison with published scores.", "Vývoj, časová validace (hlavní a citlivostní rozdělení) a srovnání s publikovanými skóre."]],
    ["5a", "Data", ["Sources of the development and evaluation data, and why they suit the question.", "Zdroje vývojových a hodnoticích dat a proč odpovídají otázce."], "done", ["data"],
      ["FNO thrombectomy registry, consecutive patients; describe how entries are made.", "Registr trombektomií FNO, konsekutivní pacienti; popsat, jak se záznamy pořizují."]],
    ["5b", "Data", ["Dates of data collection, including the end of follow-up.", "Data sběru, včetně konce sledování."], "done", ["data"], ["2005–2025; 90-day follow-up.", "2005–2025; sledování 90 dní."]],
    ["6a", "Participants", ["Setting, number and location of centres.", "Prostředí, počet a umístění center."], "done", ["about"], ["One comprehensive stroke centre (University Hospital Ostrava).", "Jedno komplexní cerebrovaskulární centrum (FN Ostrava)."]],
    ["6b", "Participants", ["Eligibility criteria.", "Kritéria zařazení."], "done", ["data"], ["Adults with a recorded 90-day mRS; every procedure type included.", "Dospělí se zaznamenaným mRS za 90 dní; zahrnuty všechny typy výkonu."]],
    ["6c", "Participants", ["Treatments received and how they were handled.", "Podaná léčba a jak s ní bylo zacházeno."], "done", ["data", "calc"],
      ["IV thrombolysis is an admission predictor; technique, stenting, angioplasty, intra-arterial lysis and anaesthesia enter at the end of the procedure.", "IV trombolýza je prediktor při přijetí; technika, stent, angioplastika, intraarteriální lýza a anestezie vstupují na konci výkonu."]],
    ["7", "Data preparation", ["Pre-processing and quality checks, and whether they were similar across sociodemographic groups.", "Předzpracování a kontroly kvality a zda byly stejné napříč sociodemografickými skupinami."], "done", ["data"],
      ["One rule set for all patients; completeness by sex and age is shown.", "Jedna sada pravidel pro všechny; úplnost podle pohlaví a věku je uvedena."]],
    ["8a", "Outcome", ["Outcome and time horizon, how and when assessed, why chosen, consistency across groups.", "Výsledek a časový horizont, jak a kdy hodnocen, proč zvolen, jednotnost napříč skupinami."], "partly", ["about", "ms"],
      ["mRS 0–2 and death at 90 days. Document how the 90-day mRS was obtained (visit or telephone, structured interview).", "mRS 0–2 a úmrtí za 90 dní. Doplnit, jak bylo mRS za 90 dní získáno (kontrola či telefon, strukturovaný rozhovor)."]],
    ["8b", "Outcome", ["Qualifications and characteristics of the outcome assessors, if judgement is needed.", "Kvalifikace a charakteristika hodnotitelů výsledku, vyžaduje-li úsudek."], "todo", ["ms"], ["Who rated the mRS (neurologist, stroke nurse; certification).", "Kdo hodnotil mRS (neurolog, sestra; certifikace)."]],
    ["8c", "Outcome", ["Any blinding of the outcome assessment.", "Případné zaslepení hodnocení výsledku."], "todo", ["ms"], ["State whether assessors knew the hospital course; most likely not blinded.", "Uvést, zda hodnotitelé znali průběh hospitalizace; nejspíš nezaslepeno."]],
    ["9a", "Predictors", ["Choice of candidate predictors and any pre-selection.", "Volba kandidátních prediktorů a případný předvýběr."], "done", ["data"],
      ["The 33 pre-procedural predictors of the 2026 analysis, then every registry item known at each later stage; columns not used are listed with reasons.", "33 předprocedurálních prediktorů z analýzy 2026 a poté každá položka registru známá v pozdější fázi; nepoužité sloupce jsou uvedeny s důvody."]],
    ["9b", "Predictors", ["Definitions of all predictors, how and when measured, any blinding.", "Definice všech prediktorů, jak a kdy měřeny, případné zaslepení."], "partly", ["data", "calc"],
      ["Definitions and timing are in the app; add how ASPECTS (which scan, reader) and NIHSS were measured.", "Definice a načasování jsou v aplikaci; doplnit, jak bylo měřeno ASPECTS (které vyšetření, hodnotitel) a NIHSS."]],
    ["9c", "Predictors", ["Qualifications of the predictor assessors, where judgement is needed.", "Kvalifikace hodnotitelů prediktorů, vyžaduje-li úsudek."], "todo", ["ms"], ["mTICI graded by the operator or a second reader? ASPECTS by radiologist or neurologist?", "Hodnotil mTICI operatér, nebo druhý hodnotitel? ASPECTS radiolog, nebo neurolog?"]],
    ["10", "Sample size", ["How the study size was arrived at and why it suffices, with any calculation.", "Jak byla stanovena velikost souboru a proč stačí, včetně výpočtu."], "partly", ["std"],
      ["Riley criteria for development and validation are computed below; report that the 33-predictor models fall short of them, the expected shrinkage, and the small 2023–2025 test.", "Kritéria Rileyho pro vývoj i validaci jsou spočtena níže; uvést, že modely s 33 prediktory je nesplňují, očekávanou kontrakci a malý test 2023–2025."]],
    ["11", "Missing data", ["How missing data were handled, with reasons for omitting data.", "Jak se zacházelo s chybějícími daty, s důvody vynechání."], "done", ["data"],
      ["No predictor values filled in before modelling (indicators, tree branches); outcome complete-case with an analysis of who lacks it.", "Žádné hodnoty prediktorů se před modelováním nedoplňují (indikátory, větve stromů); výsledek jen u úplných, s analýzou, komu chybí."]],
    ["12a", "Analysis", ["How the data were used for development and evaluation, including partitioning.", "Jak byla data použita pro vývoj a hodnocení, včetně rozdělení."], "done", ["val", "data"], ["Temporal split by treatment year, sensitivity split, nested cross-validation.", "Časové rozdělení podle roku léčby, citlivostní rozdělení, vnořená křížová validace."]],
    ["12b", "Analysis", ["How predictors were handled: functional form, rescaling, transformation.", "Zpracování prediktorů: funkční tvar, škálování, transformace."], "done", ["val"], ["Restricted cubic splines, standardisation, one-hot coding, monotone constraints.", "Restrikované kubické splajny, standardizace, one-hot kódování, monotónní omezení."]],
    ["12c", "Analysis", ["Model type, rationale, every building step including tuning, internal validation.", "Typ modelu, zdůvodnění, všechny kroky včetně ladění, interní validace."], "done", ["val"], ["Five learners, equal tuning budget on the same folds, a selection rule fixed before testing.", "Pět modelů, stejný rozpočet ladění na stejných foldech, pravidlo výběru stanovené před testováním."]],
    ["12d", "Analysis", ["Heterogeneity of parameters and performance across clusters.", "Heterogenita parametrů a výkonu napříč klastry."], "partly", ["adapt", "multi"], ["Single centre, so not applicable to centres yet (say so); eras serve as clusters (internal–external validation). Applies in the multicentre phase.", "Jedno centrum, pro centra se zatím neuplatní (uvést to); klastry jsou období (interně-externí validace). Uplatní se v multicentrické fázi."]],
    ["12e", "Analysis", ["Performance measures and plots, with rationale, and fairness measures if relevant.", "Míry a grafy výkonu se zdůvodněním a případně míry férovosti."], "done", ["val"], ["AUROC, Brier, calibration intercept, slope and O/E, calibration plots, decision curves, thresholds, subgroups.", "AUROC, Brier, kalibrační intercept, sklon a O/E, kalibrační grafy, rozhodovací křivky, prahy, podskupiny."]],
    ["12f", "Analysis", ["Any model updating, overall or for particular groups.", "Případná aktualizace modelu, celkově nebo pro skupiny."], "done", ["val", "adapt"], ["Intercept re-estimated on the recent era; six update strategies compared.", "Intercept přepočten na nedávné období; porovnáno šest strategií aktualizace."]],
    ["12g", "Analysis", ["How predictions were calculated in the evaluation (formula, code, object).", "Jak byly predikce v hodnocení počítány (vzorec, kód, objekt)."], "done", ["repo"], ["Same model objects; the browser code reproduces them exactly (parity tests).", "Stejné objekty modelů; kód v prohlížeči je reprodukuje přesně (testy shody)."]],
    ["13", "Class imbalance", ["Any class-imbalance methods, why and how, and recalibration after them.", "Metody pro nevyváženost tříd, proč a jak, a následná rekalibrace."], "done", ["ms"], ["None used (outcome rates 38% and 21%); state it.", "Nepoužity (podíly výsledku 38 % a 21 %); uvést to."]],
    ["14", "Fairness", ["Approaches to model fairness and their rationale.", "Přístupy k férovosti modelu a jejich zdůvodnění."], "partly", ["val"], ["Calibration and discrimination by sex and age are reported; no fairness correction applied; say so.", "Kalibrace a diskriminace podle pohlaví a věku jsou uvedeny; žádná korekce férovosti; uvést to."]],
    ["15", "Model output", ["The model output and any classification thresholds.", "Výstup modelu a případné klasifikační prahy."], "done", ["calc"], ["Probabilities with an uncertainty range; thresholds only described.", "Pravděpodobnosti s rozpětím nejistoty; prahy jen popsány."]],
    ["16", "Training vs evaluation", ["Differences between development and evaluation data (setting, eligibility, outcome, predictors).", "Rozdíly mezi vývojovými a hodnoticími daty (prostředí, zařazení, výsledek, prediktory)."], "done", ["data", "val"], ["Table 1 with SMD, missing data by split, outcome drift.", "Tabulka 1 se SMD, chybějící data podle rozdělení, posun výsledků."]],
    ["17", "Ethics", ["Ethics committee approval and consent or waiver.", "Schválení etickou komisí a souhlas nebo jeho prominutí."], "todo", ["ms"], ["Add the FNO ethics committee reference and the consent waiver for registry analysis.", "Doplnit číslo jednací etické komise FNO a prominutí souhlasu pro analýzu registru."]],
    ["18a", "Funding", ["Funding source and the funder's role.", "Zdroj financování a role poskytovatele."], "partly", ["about"], ["Grant SGS18/LF/2026 is listed; state the funder's role (none in design or analysis).", "Grant SGS18/LF/2026 je uveden; uvést roli poskytovatele (žádná v designu ani analýze)."]],
    ["18b", "Conflicts", ["Conflicts of interest of all authors.", "Střet zájmů všech autorů."], "todo", ["ms"], ["Declarations from every author.", "Prohlášení každého autora."]],
    ["18c", "Protocol", ["Where the protocol is, or that none was prepared.", "Kde je protokol, nebo že nebyl připraven."], "todo", ["repo"], ["Write a short analysis protocol (cohort, outcomes, learners, splits, selection rule; already fixed in config/settings.yaml) and publish it.", "Sepsat krátký protokol analýzy (kohorta, výsledky, modely, rozdělení, pravidlo výběru; už pevně v config/settings.yaml) a zveřejnit jej."]],
    ["18d", "Registration", ["Registration, or a statement that there was none.", "Registrace, nebo prohlášení, že nebyla."], "todo", ["ms"], ["Register the protocol (for example on OSF) before the next analysis round, or state not registered.", "Registrovat protokol (např. na OSF) před dalším kolem analýzy, nebo uvést, že registrován nebyl."]],
    ["18e", "Data sharing", ["Availability of the study data.", "Dostupnost dat studie."], "partly", ["registry"], ["Aggregates are public; individual data on request to FNO with ethics approval (GDPR). Write the statement.", "Souhrny jsou veřejné; individuální data na žádost ve FNO se souhlasem etické komise (GDPR). Sepsat prohlášení."]],
    ["18f", "Code sharing", ["Availability of the analysis code.", "Dostupnost analytického kódu."], "partly", ["repo"], ["Code is public on GitHub; archive the release with a DOI (Zenodo).", "Kód je veřejný na GitHubu; archivovat vydání s DOI (Zenodo)."]],
    ["19", "Patient and public involvement", ["Any patient or public involvement, or none.", "Zapojení pacientů či veřejnosti, nebo žádné."], "todo", ["calc"], ["Ask stroke survivors and relatives to review the calculator's wording, or state that there was none.", "Požádat pacienty po CMP a jejich blízké o posouzení textů kalkulačky, nebo uvést, že zapojení nebylo."]],
    ["20a", "Participants", ["Flow of participants with and without the outcome, follow-up.", "Tok účastníků s výsledkem i bez něj, sledování."], "done", ["data"], ["Flow diagram, stage cohorts and who lacks the outcome.", "Diagram toku, kohorty fází a komu chybí výsledek."]],
    ["20b", "Participants", ["Characteristics overall and per data source: dates, predictors, treatments, events, missing data; differences across groups.", "Charakteristiky celkem a podle zdroje: data, prediktory, léčba, události, chybějící data; rozdíly mezi skupinami."], "done", ["data"], ["Table 1 by split and by outcome; missing data by era, sex and age.", "Tabulka 1 podle rozdělení a výsledku; chybějící data podle období, pohlaví a věku."]],
    ["20c", "Participants", ["For evaluation, the distribution of key variables compared with the development data.", "U hodnocení srovnání rozložení klíčových proměnných s vývojovými daty."], "done", ["data"], ["SMD between development and validation years.", "SMD mezi vývojovými a validačními roky."]],
    ["21", "Model development", ["Participants and events in every analysis (development, tuning, evaluation).", "Účastníci a události v každé analýze (vývoj, ladění, hodnocení)."], "done", ["val", "data"], ["Shown with every result.", "Uvedeno u každého výsledku."]],
    ["22", "Model specification", ["The full model, so that others can make predictions and evaluate it, with any access restrictions.", "Úplný model, aby jej ostatní mohli použít a ověřit, s případnými omezeními."], "done", ["repo", "about"], ["Model bundle with fingerprint and open code; state the licence. A 2026 external validation of post-thrombectomy models could not validate any machine-learning model, because predictors were not routinely collected or code was unavailable (J NeuroInterv Surg 2026, doi:10.1136/jnis-2026-025297).", "Balík modelu s otiskem a otevřený kód; uvést licenci. Externí validace modelů po trombektomii z roku 2026 nemohla ověřit žádný model strojového učení, protože prediktory nebyly rutinně sbírány nebo nebyl dostupný kód (J NeuroInterv Surg 2026, doi:10.1136/jnis-2026-025297)."]],
    ["23a", "Performance", ["Performance with confidence intervals, including key subgroups.", "Výkon s intervaly spolehlivosti, včetně klíčových podskupin."], "done", ["val"], ["Bootstrap CIs overall and for subgroup AUROCs.", "Bootstrapové IS celkově i pro AUROC podskupin."]],
    ["23b", "Performance", ["Heterogeneity of performance across clusters, if examined.", "Heterogenita výkonu napříč klastry, byla-li zkoumána."], "partly", ["adapt", "multi"], ["Across eras (forest plot, I²); across centres later (TRIPOD-Cluster).", "Napříč obdobími (forest plot, I²); napříč centry později (TRIPOD-Cluster)."]],
    ["24", "Model updating", ["Results of any updating, with the updated model and its performance.", "Výsledky aktualizace, s aktualizovaným modelem a jeho výkonem."], "done", ["val", "adapt"], ["Original vs recalibrated; the published model is the updated one.", "Původní vs rekalibrovaný; zveřejněn je aktualizovaný model."]],
    ["25", "Interpretation", ["Overall interpretation, including fairness, against the objectives and earlier studies.", "Celková interpretace, včetně férovosti, vzhledem k cílům a dřívějším studiím."], "todo", ["ms"], ["Manuscript discussion.", "Diskuse rukopisu."]],
    ["26", "Limitations", ["Limitations (sample, size, overfitting, missing data) and their effects.", "Omezení (soubor, velikost, přeučení, chybějící data) a jejich dopady."], "partly", ["about"], ["Listed on About; expand in the manuscript.", "Uvedena v O projektu; rozvést v rukopisu."]],
    ["27a", "Usability", ["How poor-quality or missing inputs should be assessed and handled in use.", "Jak při použití posoudit a zpracovat nekvalitní nebo chybějící vstupy."], "done", ["calc"], ["Unknown inputs are averaged over registry values and listed under the estimate.", "Neznámé vstupy se průměrují přes hodnoty registru a jsou vypsány pod odhadem."]],
    ["27b", "Usability", ["Whether users handle the inputs and what expertise they need.", "Zda uživatelé zadávají vstupy a jakou odbornost potřebují."], "partly", ["calc"], ["Inputs are routine stroke-care items entered by a physician; state the expertise needed (NIHSS, mTICI).", "Vstupy jsou běžné položky péče o CMP zadávané lékařem; uvést potřebnou odbornost (NIHSS, mTICI)."]],
    ["27c", "Usability", ["Next steps for research, with a view to applicability and generalisability.", "Další výzkum s ohledem na použitelnost a zobecnitelnost."], "done", ["road", "multi"], ["External validation, federated learning, prospective silent validation.", "Externí validace, federované učení, prospektivní tichá validace."]],
  ];

  const ABSTRACT = [
    ["Title", ["Development or evaluation of a prediction model, population, outcome", "Vývoj či hodnocení predikčního modelu, populace, výsledek"]],
    ["Background", ["Healthcare context and rationale", "Klinický kontext a zdůvodnění"]],
    ["Objectives", ["Development, validation or both", "Vývoj, validace nebo obojí"]],
    ["Methods: data", ["Sources of data", "Zdroje dat"]],
    ["Methods: participants", ["Eligibility criteria and setting", "Kritéria zařazení a prostředí"]],
    ["Methods: outcome", ["Outcome and time horizon", "Výsledek a časový horizont"]],
    ["Methods: model", ["Model type, building steps, internal validation", "Typ modelu, kroky tvorby, interní validace"]],
    ["Methods: evaluation", ["Measures of performance (and fairness)", "Míry výkonu (a férovosti)"]],
    ["Results: participants", ["Number of participants and outcome events", "Počet účastníků a událostí"]],
    ["Results: model", ["Predictors in the final model", "Prediktory výsledného modelu"]],
    ["Results: performance", ["Performance estimates with confidence intervals", "Odhady výkonu s intervaly spolehlivosti"]],
    ["Discussion", ["Overall interpretation", "Celková interpretace"]],
    ["Registration", ["Registry and number, or not registered", "Registr a číslo, nebo neregistrováno"]],
  ];

  // PROBAST+AI (Moons et al., BMJ 2025;388:e082505), our self-assessment; paraphrased domains
  const PROBAST = [
    { d: ["Participants and data sources", "Účastníci a zdroje dat"],
      dev: ["low", ["Consecutive patients of a prospective registry; all procedure types included; only patients without a 90-day mRS excluded.", "Konsekutivní pacienti prospektivního registru; zahrnuty všechny typy výkonu; vyřazeni jen pacienti bez mRS za 90 dní."]],
      ev: ["some", ["Same registry in later years, but the 90-day mRS is missing for about one in five recent patients.", "Stejný registr v pozdějších letech, ale mRS za 90 dní chybí přibližně u každého pátého nedávného pacienta."]],
      app: ["some", ["One Czech comprehensive stroke centre; applicability elsewhere needs external validation.", "Jedno české komplexní cerebrovaskulární centrum; použitelnost jinde vyžaduje externí validaci."]] },
    { d: ["Predictors", "Prediktory"],
      dev: ["low", ["Registry definitions; each stage uses only what is known at that moment; recorded before the outcome.", "Definice registru; každá fáze používá jen to, co je v daném okamžiku známo; zaznamenáno před výsledkem."]],
      ev: ["some", ["Some items were recorded less often in recent years (smoking, 24-h NIHSS) or only from 2016 (ASPECTS).", "Některé položky byly v posledních letech zaznamenávány méně (kouření, NIHSS za 24 h) nebo až od roku 2016 (ASPECTS)."]],
      app: ["low", ["Routine items available in any thrombectomy centre.", "Běžné položky dostupné v každém centru trombektomie."]] },
    { d: ["Outcome", "Výsledek"],
      dev: ["some", ["Standard outcome (90-day mRS), but how it was assessed and whether assessors knew the hospital course is not documented.", "Standardní výsledek (mRS za 90 dní), ale způsob hodnocení a zda hodnotitelé znali průběh hospitalizace není zdokumentován."]],
      ev: ["some", ["As in development.", "Jako u vývoje."]],
      app: ["low", ["mRS 0–2 and death at 90 days are the usual trial outcomes.", "mRS 0–2 a úmrtí za 90 dní jsou obvyklé výsledky studií."]] },
    { d: ["Analysis", "Analýza"],
      dev: ["some", ["No dichotomised predictors, no imputation before splitting, penalised regression, equal tuning for all learners, nested cross-validation, calibration assessed, no data leakage across stages, no class rebalancing; but the 33-predictor models have fewer events per parameter than the Riley criteria ask for.", "Žádné dichotomizované prediktory, žádná imputace před rozdělením, penalizovaná regrese, stejné ladění všech modelů, vnořená křížová validace, hodnocena kalibrace, žádný únik dat mezi fázemi, žádné převažování tříd; modely s 33 prediktory však mají méně událostí na parametr, než požadují kritéria Rileyho."]],
      ev: ["some", ["Temporal test with fewer than 100 good outcomes (wide intervals), hence the sensitivity split; calibration, discrimination and net benefit all reported.", "Časový test s méně než 100 dobrými výsledky (široké intervaly), proto citlivostní rozdělení; uvedena kalibrace, diskriminace i čistý přínos."]],
      app: null },
  ];

  const OTHERS = [
    { n: "TRIPOD+AI", ref: "Collins et al., BMJ 2024;385:e078378", what: ["Reporting of prediction models built with regression or machine learning", "Reportování predikčních modelů z regrese či strojového učení"], when: ["Now (the manuscript)", "Nyní (rukopis)"], st: "partly" },
    { n: "PROBAST+AI", ref: "Moons et al., BMJ 2025;388:e082505", what: ["Quality, risk of bias and applicability of prediction model studies", "Kvalita, riziko zkreslení a použitelnost studií predikčních modelů"], when: ["Now: self-check; later: independent reviewer", "Nyní: vlastní kontrola; později nezávislý hodnotitel"], st: "partly" },
    { n: "RECORD", ref: "Benchimol et al., PLoS Med 2015;12:e1001885", what: ["Studies of routinely collected data (registries): codes, data cleaning, linkage, access", "Studie rutinně sbíraných dat (registry): kódy, čištění, propojení, přístup"], when: ["Now", "Nyní"], st: "partly", note: ["Codes and cleaning are on the Registry page; add the data-access statement.", "Kódy a čištění jsou na stránce Registr; doplnit prohlášení o přístupu k datům."] },
    { n: "TRIPOD-Cluster", ref: "Debray et al., BMJ 2023;380:e071018", what: ["Prediction models from clustered data (several centres)", "Predikční modely z klastrovaných dat (více center)"], when: ["Multicentre and federated phase", "Multicentrická a federovaná fáze"], st: "later" },
    { n: "DECIDE-AI", ref: "Vasey et al., Nat Med 2022;28:924–33", what: ["Early clinical evaluation of AI decision support with real users", "Časné klinické hodnocení AI podpory rozhodování se skutečnými uživateli"], when: ["Before any clinical pilot", "Před jakýmkoli klinickým pilotem"], st: "later" },
    { n: "SPIRIT-AI / CONSORT-AI", ref: "Cruz Rivera et al.; Liu et al., Nat Med 2020", what: ["Protocols and reports of randomised trials of AI interventions", "Protokoly a zprávy randomizovaných studií AI intervencí"], when: ["Only if a trial is run", "Jen při randomizované studii"], st: "later" },
    { n: "FUTURE-AI", ref: "Lekadir et al., BMJ 2025;388:e081554", what: ["Principles of trustworthy, deployable AI in healthcare", "Zásady důvěryhodné a nasaditelné AI ve zdravotnictví"], when: ["Roadmap", "Plán rozvoje"], st: "later" },
    { n: "CLAIM, TRIPOD-LLM", ref: "", what: ["Imaging AI; large language models", "AI v zobrazování; velké jazykové modely"], when: ["Not applicable", "Neuplatní se"], st: "na" },
    { n: ["EU MDR 2017/745, rule 11", "EU MDR 2017/745, pravidlo 11"], ref: "MDCG 2019-11", what: ["Software that informs diagnostic or therapeutic decisions is a medical device of class IIa or higher", "Software, který informuje diagnostická či terapeutická rozhodnutí, je zdravotnickým prostředkem třídy IIa či vyšší"], when: ["Only if clinical use is pursued", "Jen při snaze o klinické použití"], st: "na",
      note: ["An individual prognosis offered for clinical use would qualify as medical device software (class IIa or higher). MTOP is labelled for research and teaching and avoids recommendation wording; clinical use needs CE marking, or for FNO alone the in-house route of MDR Art. 5(5).", "Individuální prognóza nabízená ke klinickému použití by byla softwarem zdravotnického prostředku (třída IIa či vyšší). MTOP je označen pro výzkum a výuku a vyhýbá se doporučujícím formulacím; klinické použití vyžaduje označení CE, nebo jen pro FNO cestu vlastní výroby podle čl. 5 odst. 5 MDR."] },
    { n: ["EU AI Act 2024/1689", "EU AI Act 2024/1689"], ref: ["amended by Regulation (EU) 2026/1744", "ve znění nařízení (EU) 2026/1744"], what: ["AI in a medical device that needs a notified body is high-risk; for such products the obligations apply from 2 August 2028", "AI ve zdravotnickém prostředku s notifikovanou osobou je vysoce riziková; pro takové výrobky platí povinnosti od 2. 8. 2028"], when: ["Only if clinical use is pursued", "Jen při snaze o klinické použití"], st: "na",
      note: ["AI developed and used only for scientific research is outside the Act.", "AI vyvinutá a používaná jen pro vědecký výzkum pod nařízení nespadá."] },
    { n: "GDPR", ref: "", what: ["Personal data", "Osobní údaje"], when: ["Now", "Nyní"], st: "done", note: ["Only aggregates and models leave FNO (groups under 10 hidden, tree leaves of at least 20); a data protection impact assessment before the federated network goes live.", "FNO opouštějí jen souhrny a modely (skupiny pod 10 skryty, listy stromů alespoň 20); před spuštěním federované sítě posouzení vlivu na ochranu údajů."] },
    { n: ["Journal policies", "Pravidla časopisů"], ref: ["e.g. Scientific Reports", "např. Scientific Reports"], what: ["Data and code availability statements; the completed checklist as a supplement", "Prohlášení o dostupnosti dat a kódu; vyplněný checklist jako příloha"], when: ["At submission", "Při podání"], st: "partly" },
  ];

  const ACTIONS = [
    [["Ethics approval and consent waiver", "Souhlas etické komise a prominutí souhlasu"], "17", ["Record the FNO ethics committee reference for the registry analysis.", "Doplnit číslo jednací etické komise FNO pro analýzu registru."]],
    [["Protocol and registration", "Protokol a registrace"], "18c, 18d", ["Publish a short analysis protocol (the settings already fixed in config/settings.yaml) and register it, for example on OSF, before the next analysis round.", "Zveřejnit krátký protokol analýzy (nastavení už pevně v config/settings.yaml) a registrovat jej, např. na OSF, před dalším kolem analýzy."]],
    [["How the outcome was assessed", "Jak byl hodnocen výsledek"], "8a–8c", ["Who obtained the 90-day mRS, how (visit or telephone, structured), and whether they knew the hospital course.", "Kdo zjišťoval mRS za 90 dní, jak (kontrola či telefon, strukturovaně) a zda znal průběh hospitalizace."]],
    [["How predictors were assessed", "Jak byly hodnoceny prediktory"], "9b, 9c", ["NIHSS certification, ASPECTS reading (scan, reader), mTICI grading (operator or second reader).", "Certifikace NIHSS, hodnocení ASPECTS (vyšetření, hodnotitel), hodnocení mTICI (operatér či druhý hodnotitel)."]],
    [["Archive code and model with a DOI", "Archivovat kód a model s DOI"], "18f, 22", ["Tag the release, archive it on Zenodo, state the licence; add the data-availability statement.", "Označit vydání, archivovat na Zenodu, uvést licenci; doplnit prohlášení o dostupnosti dat."]],
    [["Fairness and inequalities", "Férovost a nerovnosti"], "3c, 14", ["Background on sex and age differences; report subgroup calibration (on the Validation page) and say that no fairness correction was applied.", "Podklady k rozdílům podle pohlaví a věku; uvést kalibraci v podskupinách (stránka Validace) a že nebyla použita korekce férovosti."]],
    [["Patient and public involvement", "Zapojení pacientů a veřejnosti"], "19", ["Ask stroke survivors and relatives to review the calculator's wording and the prognosis display, or state that there was none.", "Požádat pacienty po CMP a jejich blízké o posouzení textů kalkulačky a zobrazení prognózy, nebo uvést, že zapojení nebylo."]],
    [["Funding and conflicts of interest", "Financování a střet zájmů"], "18a, 18b", ["The funder's role and every author's declaration.", "Role poskytovatele a prohlášení každého autora."]],
    [["Write the manuscript items", "Sepsat položky rukopisu"], "1, 2, 25, 26", ["Title, abstract by TRIPOD+AI for Abstracts, interpretation, limitations.", "Název, abstrakt podle TRIPOD+AI pro abstrakty, interpretace, omezení."]],
    [["Independent PROBAST+AI assessment", "Nezávislé hodnocení PROBAST+AI"], "–", ["By someone outside the team.", "Někým mimo tým."]],
    [["External validation", "Externí validace"], "27c", ["Partner centres can validate in the browser now (Multicentre).", "Partnerská centra mohou validovat v prohlížeči už nyní (Multicentrum)."]],
  ];

  const STATUS = {
    done: ["Addressed", "Splněno", "st-live"], partly: ["Partly", "Částečně", "st-beta"], todo: ["To do", "Zbývá", "st-warn"],
    na: ["Not applicable", "Neuplatní se", "st-planned"], later: ["Later phase", "Pozdější fáze", "st-progress"],
  };
  const RISK = { low: ["low concern", "nízké riziko", "st-live"], some: ["some concerns", "určité obavy", "st-beta"], high: ["high concern", "vysoké riziko", "st-warn"] };
  const chip = (map, k) => { const s = map[k]; return h("span", { class: `status ${s[2]}` }, h("i", { "aria-hidden": "true" }), tx(s[0], s[1])); };
  const two = (p) => (Array.isArray(p) ? tx(p[0], p[1]) : p);

  function whereLinks(keys) {
    return keys.map((k) => {
      const w = W[k];
      if (!w) return null;
      if (!w[0]) return h("span", { class: "where", text: tx(w[1], w[2]) });
      return h("a", { href: `#${w[0]}`, class: "where", onclick: (e) => { e.preventDefault(); if (w[3]) Object.assign(S, w[3]); A.goTo(w[0]); window.scrollTo(0, 0); }, text: tx(w[1], w[2]) });
    });
  }

  function head(title, sub) {
    return h("div", { class: "chart-head" }, h("div", null, h("div", { class: "chart-title", text: title }), sub ? h("div", { class: "chart-sub", text: sub }) : null));
  }

  function tripodCard(root) {
    const card = h("div", { class: "card card-pad block", id: "std-tripod" });
    const counts = {};
    for (const r of TRIPOD) counts[r[3]] = (counts[r[3]] || 0) + 1;
    card.appendChild(head(tx("TRIPOD+AI checklist", "Checklist TRIPOD+AI"),
      tx(`27 items, ${TRIPOD.length} sub-items · addressed ${counts.done || 0}, partly ${counts.partly || 0}, to do ${counts.todo || 0}`,
        `27 položek, ${TRIPOD.length} podpoložek · splněno ${counts.done || 0}, částečně ${counts.partly || 0}, zbývá ${counts.todo || 0}`)));
    const f = S.repFilter;
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": tx("Show", "Zobrazit") },
      [["all", tx("All", "Vše")], ["todo", tx("To do", "Zbývá")], ["partly", tx("Partly", "Částečně")], ["done", tx("Addressed", "Splněno")]].map(([k, lbl]) =>
        h("button", { type: "button", "aria-pressed": String(f === k), onclick: () => { S.repFilter = k; const y = window.scrollY; render(root); window.scrollTo(0, y); } }, lbl)));
    card.appendChild(h("div", { class: "controls" }, h("span", { text: tx("Show", "Zobrazit") }), seg));
    const rows = [];
    let sec = null;
    for (const r of TRIPOD) {
      if (f !== "all" && r[3] !== f) continue;
      if (r[1] !== sec) {
        sec = r[1];
        rows.push(h("tr", { class: "sep" }, h("th", { colspan: 5, text: tx(sec, SECTION_CS[sec] || sec) })));
      }
      rows.push(h("tr", null, h("th", { scope: "row", class: "num", text: r[0] }), h("td", { text: two(r[2]) }), h("td", null, chip(STATUS, r[3])),
        h("td", { class: "wh" }, whereLinks(r[4])), h("td", { class: "muted", text: two(r[5]) })));
    }
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data std" },
      h("thead", null, h("tr", null, [tx("Item", "Položka"), tx("Asks for (paraphrased)", "Požaduje (parafráze)"), tx("Status", "Stav"), tx("Where", "Kde"), tx("What to do", "Co udělat")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows))));
    card.appendChild(h("p", { class: "note", text: tx(
      "Items paraphrased from the TRIPOD+AI statement (Collins et al., BMJ 2024); use the official checklist for the submission and add manuscript page numbers. “Addressed” means the analysis exists and is shown in MTOP; it still has to be written up.",
      "Položky jsou parafrázovány podle TRIPOD+AI (Collins a kol., BMJ 2024); pro podání použijte oficiální checklist a doplňte čísla stran rukopisu. „Splněno“ znamená, že analýza existuje a je v MTOP zobrazena; stále je třeba ji sepsat.") }));
    return card;
  }
  const SECTION_CS = {
    Title: "Název", Abstract: "Abstrakt", Background: "Úvod", Objectives: "Cíle", Data: "Data", Participants: "Účastníci", "Data preparation": "Příprava dat",
    Outcome: "Výsledek", Predictors: "Prediktory", "Sample size": "Velikost souboru", "Missing data": "Chybějící data", Analysis: "Analýza",
    "Class imbalance": "Nevyváženost tříd", Fairness: "Férovost", "Model output": "Výstup modelu", "Training vs evaluation": "Trénink vs hodnocení",
    Ethics: "Etika", Funding: "Financování", Conflicts: "Střet zájmů", Protocol: "Protokol", Registration: "Registrace", "Data sharing": "Sdílení dat",
    "Code sharing": "Sdílení kódu", "Patient and public involvement": "Zapojení pacientů a veřejnosti", "Model development": "Vývoj modelu",
    "Model specification": "Specifikace modelu", Performance: "Výkon", "Model updating": "Aktualizace modelu", Interpretation: "Interpretace",
    Limitations: "Omezení", Usability: "Použitelnost",
  };

  function abstractCard() {
    const card = h("div", { class: "card card-pad" });
    card.appendChild(head(tx("TRIPOD+AI for Abstracts", "TRIPOD+AI pro abstrakty"), tx("13 items, all for the manuscript abstract", "13 položek, všechny pro abstrakt rukopisu")));
    card.appendChild(h("ol", { class: "std-list" }, ABSTRACT.map(([k, d]) => h("li", null, h("b", { text: tx(k, ABS_CS[k] || k) }), ` · ${two(d)}`))));
    return card;
  }
  const ABS_CS = { Title: "Název", Background: "Úvod", Objectives: "Cíle", "Methods: data": "Metody: data", "Methods: participants": "Metody: účastníci",
    "Methods: outcome": "Metody: výsledek", "Methods: model": "Metody: model", "Methods: evaluation": "Metody: hodnocení", "Results: participants": "Výsledky: účastníci",
    "Results: model": "Výsledky: model", "Results: performance": "Výsledky: výkon", Discussion: "Diskuse", Registration: "Registrace" };

  function probastCard() {
    const card = h("div", { class: "card card-pad" });
    card.appendChild(head(tx("PROBAST+AI self-assessment", "Vlastní hodnocení PROBAST+AI"),
      tx("Development: quality · evaluation: risk of bias · applicability. Our own rating; an independent reviewer should repeat it.", "Vývoj: kvalita · hodnocení: riziko zkreslení · použitelnost. Vlastní hodnocení; měl by jej zopakovat nezávislý hodnotitel.")));
    const rows = PROBAST.map((p) => h("tr", null, h("th", { scope: "row", text: two(p.d) }),
      h("td", null, chip(RISK, p.dev[0]), h("div", { class: "muted small", text: two(p.dev[1]) })),
      h("td", null, chip(RISK, p.ev[0]), h("div", { class: "muted small", text: two(p.ev[1]) })),
      h("td", null, p.app ? chip(RISK, p.app[0]) : "–", p.app ? h("div", { class: "muted small", text: two(p.app[1]) }) : null)));
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data std" },
      h("thead", null, h("tr", null, [tx("Domain", "Doména"), tx("Development (quality)", "Vývoj (kvalita)"), tx("Temporal validation (risk of bias)", "Časová validace (riziko zkreslení)"), tx("Applicability", "Použitelnost")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows))));
    card.appendChild(h("p", { class: "note", text: tx(
      "Overall: some concerns. Development is careful but has fewer events per parameter than recommended for 33 predictors; the temporal validation is small and recent outcomes are partly missing. Applicability beyond FNO is unknown until external validation. Domains paraphrased from PROBAST+AI (Moons et al., BMJ 2025; CC BY-NC 4.0).",
      "Celkově: určité obavy. Vývoj je pečlivý, ale pro 33 prediktorů má méně událostí na parametr, než se doporučuje; časová validace je malá a nedávné výsledky zčásti chybějí. Použitelnost mimo FNO není známa do externí validace. Domény parafrázovány podle PROBAST+AI (Moons a kol., BMJ 2025; CC BY-NC 4.0).") }));
    return card;
  }

  function sampleCard() {
    const card = h("div", { class: "card card-pad block", id: "std-size" });
    card.appendChild(head(tx("Sample size (TRIPOD+AI item 10)", "Velikost souboru (TRIPOD+AI položka 10)"),
      tx("Minimum sample size by the Riley criteria, for the reference model (regression with splines)", "Minimální velikost souboru podle kritérií Rileyho, pro referenční model (regrese se splajny)")));
    if (!R) { card.appendChild(h("p", { class: "note", text: tx("Computed after training (python -m mtop reporting).", "Počítá se po trénování (python -m mtop reporting).") })); return card; }
    const ref = A.mname(R.reference || A.REF);
    const D = R.development || {};
    const tg = (k) => (k === "good_outcome" ? "mRS 0–2" : tx("death", "úmrtí"));
    const devRows = [];
    for (const st of A.STAGES) for (const k of A.TARGETS) {
      const r = (D[st] || {})[k];
      if (!r) continue;
      devRows.push(h("tr", null, h("th", { scope: "row", text: `${A.stageLabel(st)} · ${tg(k)}` }), h("td", { class: "num", text: num(r.p) }),
        h("td", { class: "num", text: A.pct(r.prevalence, 0) }), h("td", { class: "num", text: num(r.r2_cs, 3) }),
        h("td", { class: "num", text: `${num(r.n_i)} / ${num(r.n_ii)} / ${num(r.n_iii)}` }), h("td", { class: "num", text: num(r.n_required) }),
        h("td", { class: "num", text: `${num(r.n_available)} (${num(r.events_available)})` }), h("td", { class: "num", text: num(r.epp_available, 1) }),
        h("td", { class: "num", text: num(r.shrinkage_expected, 2) }), h("td", null, chip(STATUS, r.met ? "done" : "todo"))));
    }
    card.appendChild(h("div", { class: "chart-sub", style: "margin:4px 0 6px", text: tx(`Development (Riley et al., BMJ 2020): ${ref}`, `Vývoj (Riley a kol., BMJ 2020): ${ref}`) }));
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
      h("thead", null, h("tr", null, [tx("Stage · outcome", "Fáze · výsledek"), tx("Parameters", "Parametry"), tx("Outcome rate", "Podíl výsledku"), "R² (Cox–Snell)",
        tx("n for (i) / (ii) / (iii)", "n pro (i) / (ii) / (iii)"), tx("n needed", "potřebné n"), tx("n available (events)", "dostupné n (události)"), tx("Events per parameter", "Události na parametr"), tx("Expected shrinkage", "Očekávaná kontrakce"), tx("Met", "Splněno")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, devRows))));
    const VS = R.validation || {};
    for (const [label, per] of Object.entries(VS)) {
      const rows = [];
      for (const st of A.STAGES) for (const k of A.TARGETS) {
        const r = (per[st] || {})[k];
        if (!r) continue;
        rows.push(h("tr", null, h("th", { scope: "row", text: `${A.stageLabel(st)} · ${tg(k)}` }), h("td", { class: "num", text: A.pct(r.prevalence, 0) }), h("td", { class: "num", text: num(r.c, 2) }),
          h("td", { class: "num", text: `${num(r.n_oe)} / ${num(r.n_slope)} / ${num(r.n_c)}` }), h("td", { class: "num", text: `${num(r.n_required)} (${num(r.events_required)})` }),
          h("td", { class: "num", text: `${num(r.n_available)} (${num(r.events_available)})` }), h("td", null, chip(STATUS, r.met ? "done" : "todo"))));
      }
      card.appendChild(h("div", { class: "chart-sub", style: "margin:14px 0 6px", text: tx(`Validation ${A.yearsOf(label)} (Riley et al., Stat Med 2021)`, `Validace ${A.yearsOf(label)} (Riley a kol., Stat Med 2021)`) }));
      card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
        h("thead", null, h("tr", null, [tx("Stage · outcome", "Fáze · výsledek"), tx("Outcome rate", "Podíl výsledku"), "C", tx("n for O/E / slope / C", "n pro O/E / sklon / C"), tx("n needed (events)", "potřebné n (události)"), tx("n available (events)", "dostupné n (události)"), tx("Met", "Splněno")].map((x) => h("th", { scope: "col", text: x })))),
        h("tbody", null, rows))));
    }
    card.appendChild(h("p", { class: "note", text: tx(
      "Development criteria: (i) expected shrinkage at least 0.9, (ii) optimism in Nagelkerke R² at most 0.05, (iii) overall risk estimated within ±0.05; parameters count every spline term, category and missing-value indicator; R² from out-of-fold predictions. Validation criteria: 95% CI widths of 0.2 for O/E and for the calibration slope and 0.1 for the C-statistic, using the reference model's observed performance. The criteria assume unpenalised maximum likelihood; MTOP's regressions are penalised (L2, chosen by cross-validation), which shrinks coefficients and reduces the overfitting the criteria guard against, but with 33 predictors before the procedure the development data fall short of criterion (i), so the expected shrinkage is reported and calibration is checked on new years. The flexible learners (trees, neural network) need more data than a regression; where criteria are not met, intervals are wide and are reported as such. Values are approximate; pmsampsize and pmvalsampsize give the same criteria.",
      "Kritéria vývoje: (i) očekávaná kontrakce alespoň 0,9, (ii) optimismus Nagelkerkeho R² nejvýše 0,05, (iii) celkové riziko odhadnuto s přesností ±0,05; parametry zahrnují každý člen splajnu, kategorii i indikátor chybění; R² z predikcí mimo fold. Kritéria validace: šířka 95% IS 0,2 pro O/E i kalibrační sklon a 0,1 pro C-statistiku, s pozorovaným výkonem referenčního modelu. Kritéria předpokládají nepenalizovanou maximální věrohodnost; regrese MTOP jsou penalizované (L2, volba křížovou validací), což koeficienty smršťuje a omezuje přeučení, proti němuž kritéria chrání, ale s 33 prediktory před výkonem vývojová data kritérium (i) nesplňují, proto je uvedena očekávaná kontrakce a kalibrace se ověřuje na nových letech. Flexibilní modely (stromy, neuronová síť) potřebují více dat než regrese; kde kritéria splněna nejsou, jsou intervaly široké a tak jsou i uvedeny. Hodnoty jsou přibližné; stejná kritéria počítají pmsampsize a pmvalsampsize.") }));
    return card;
  }

  function othersCard() {
    const card = h("div", { class: "card card-pad block", id: "std-other" });
    card.appendChild(head(tx("Other standards and regulation", "Další standardy a regulace"), tx("What applies now, what applies later and what does not apply", "Co platí nyní, co později a co se neuplatní")));
    const rows = OTHERS.map((o) => h("tr", null, h("th", { scope: "row" }, h("b", { text: two(o.n) }), o.ref ? h("div", { class: "muted small", text: two(o.ref) }) : null),
      h("td", { text: two(o.what) }), h("td", { text: two(o.when) }), h("td", null, chip(STATUS, o.st)), h("td", { class: "muted", text: o.note ? two(o.note) : "" })));
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data std" },
      h("thead", null, h("tr", null, [tx("Standard", "Standard"), tx("Covers", "Pokrývá"), tx("When", "Kdy"), tx("Status", "Stav"), tx("Note", "Poznámka")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows))));
    card.appendChild(h("p", { class: "note", text: tx(
      "Regulatory notes are orientation, not legal advice. The AI Act dates are those of Regulation (EU) 2026/1744 (the 2026 “digital omnibus”), which moved the obligations for high-risk AI in regulated products such as medical devices from 2 August 2027 to 2 August 2028.",
      "Regulatorní poznámky slouží k orientaci, nejde o právní radu. Data AI Act jsou podle nařízení (EU) 2026/1744 („digitální omnibus“ z roku 2026), které posunulo povinnosti pro vysoce rizikovou AI v regulovaných výrobcích, jako jsou zdravotnické prostředky, z 2. 8. 2027 na 2. 8. 2028.") }));
    return card;
  }

  function actionsCard() {
    const card = h("div", { class: "card card-pad block", id: "std-actions" });
    card.appendChild(head(tx("What is still to do, in order", "Co zbývá, podle pořadí"), tx("Most of it is documentation the study team holds", "Většinou jde o dokumentaci, kterou má řešitelský tým")));
    card.appendChild(h("ol", { class: "std-actions" }, ACTIONS.map(([t, item, d]) => h("li", null, h("b", { text: two(t) }), item !== "–" ? h("span", { class: "muted", text: ` · TRIPOD+AI ${item}` }) : null, h("p", { text: two(d) })))));
    return card;
  }

  function render(root) {
    root.replaceChildren();
    const done = TRIPOD.filter((r) => r[3] === "done").length;
    root.appendChild(h("div", { class: "view-head" }, h("div", null,
      h("h2", { text: tx("Reporting standards", "Standardy reportování") }),
      h("p", { text: tx(
        "MTOP is a prediction model study, so it is reported to TRIPOD+AI and checked against PROBAST+AI. Below: every TRIPOD+AI item with where MTOP addresses it and what is left, the abstract checklist, a risk-of-bias self-assessment, the sample-size criteria, the standards for later phases, and the to-do list.",
        "MTOP je studie predikčního modelu, proto se reportuje podle TRIPOD+AI a kontroluje podle PROBAST+AI. Níže: každá položka TRIPOD+AI s tím, kde ji MTOP pokrývá a co zbývá, checklist pro abstrakt, vlastní hodnocení rizika zkreslení, kritéria velikosti souboru, standardy pro další fáze a seznam úkolů.") }))));
    root.appendChild(h("div", { class: "kpis" },
      h("div", { class: "kpi" }, h("div", { class: "l", text: tx("TRIPOD+AI sub-items addressed", "Splněné podpoložky TRIPOD+AI") }), h("div", { class: "v", text: `${done} / ${TRIPOD.length}` }), h("div", { class: "s", text: tx("analysis done, write-up pending", "analýza hotová, čeká na sepsání") })),
      h("div", { class: "kpi" }, h("div", { class: "l", text: tx("Partly", "Částečně") }), h("div", { class: "v", text: String(TRIPOD.filter((r) => r[3] === "partly").length) })),
      h("div", { class: "kpi" }, h("div", { class: "l", text: tx("To do", "Zbývá") }), h("div", { class: "v", text: String(TRIPOD.filter((r) => r[3] === "todo").length) }), h("div", { class: "s", text: tx("mostly documentation", "většinou dokumentace") })),
      h("div", { class: "kpi" }, h("div", { class: "l", text: tx("Risk of bias (self-assessed)", "Riziko zkreslení (vlastní)") }), h("div", { class: "v", text: tx("some concerns", "určité obavy") }), h("div", { class: "s", text: tx("small test, missing outcomes", "malý test, chybějící výsledky") }))));
    root.appendChild(actionsCard());
    root.appendChild(tripodCard(root));
    root.appendChild(h("div", { class: "grid-2 block" }, abstractCard(), probastCard()));
    root.appendChild(sampleCard());
    root.appendChild(othersCard());
  }

  A.registerView("reporting", () => tx("Standards", "Standardy"), render);
})();
