
-- Table: siplca_bounds
CREATE TABLE siplca_bounds ( 
    track_id   TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);


-- Table: levy_bounds
CREATE TABLE levy_bounds ( 
    track_id   TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);


-- Table: olda_bounds
CREATE TABLE olda_bounds ( 
    track_id   TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);


-- Table: boundaries
CREATE TABLE boundaries ( 
    algo_id    TEXT,
    ds_name    TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);


-- Table: serra_bounds
CREATE TABLE serra_bounds ( 
    track_id   TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);


-- Table: foote_bounds
CREATE TABLE foote_bounds ( 
    track_id   TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);

-- Table: mma_bounds
CREATE TABLE mma_bounds ( 
    track_id   TEXT,
    F05        REAL,
    P05        REAL,
    R05        REAL,
    F3         REAL,
    P3         REAL,
    R3         REAL,
    D          REAL,
    annot_beat INTEGER,
    feature    TEXT,
    add_params TEXT,
    trim       INT 
);
