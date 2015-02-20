-- phpMyAdmin SQL Dump
-- version 4.0.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jul 07, 2014 at 05:31 AM
-- Server version: 5.5.33
-- PHP Version: 5.5.3

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";

--
-- Database: `urinieto_boundaries_experiment2`
--

-- --------------------------------------------------------

--
-- Table structure for table `excerpts`
--

CREATE TABLE `excerpts` (
  `id` int(2) DEFAULT NULL,
  `track_id` varchar(61) DEFAULT NULL,
  `v1F` decimal(11,10) DEFAULT NULL,
  `v1P` decimal(11,10) DEFAULT NULL,
  `v1R` int(1) DEFAULT NULL,
  `v3F` decimal(11,10) DEFAULT NULL,
  `v3P` int(1) DEFAULT NULL,
  `v3R` decimal(11,10) DEFAULT NULL,
  `v1Results` int(10) NOT NULL DEFAULT '0',
  `v2Results` int(10) NOT NULL DEFAULT '0',
  `v3Results` int(10) NOT NULL DEFAULT '0',
  `totalResults` int(10) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `excerpts`
--

INSERT INTO `excerpts` (`id`, `track_id`, `v1F`, `v1P`, `v1R`, `v3F`, `v3P`, `v3R`, `v1Results`, `v2Results`, `v3Results`, `totalResults`) VALUES
(0, 'Cerulean_Astor_Piazzolla-El_Desbande.mp3', 0.9166666667, 0.8461538462, 1, 0.9000000000, 1, 0.8181818182, 0, 0, 0, 0),
(1, 'Cerulean_Dream_Theater-6:00:00.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(2, 'Cerulean_Eddie_Palmieri-Adoracion.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(3, 'Cerulean_Elvis_Presley-Heartbreak_Hotel.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(4, 'Cerulean_English_Chamber_Orchestra_&_Raymond_Leppard-Cano.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(5, 'Cerulean_Flogging_Molly-The_Light_of_a_Fading_Star.mp3', 0.8750000000, 0.7777777778, 1, 0.8333333333, 1, 0.7142857143, 0, 0, 0, 0),
(6, 'Cerulean_Iron_&_Wine-Summer_In_Savannah.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(7, 'Cerulean_M.I.A.-Boyz.mp3', 0.9166666667, 0.8461538462, 1, 0.9000000000, 1, 0.8181818182, 0, 0, 0, 0),
(8, 'Cerulean_Meshuggah-Future_Breed_Machine.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(9, 'Cerulean_Metallica-Harvester_of_Sorrow.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(10, 'Cerulean_Metallica-Master_of_Puppets.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(11, 'Cerulean_Rush-Natural_Science.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(12, 'Cerulean_The_Dillinger_Escape_Plan-Sugar_Coated_Sour.mp3', 0.9230769231, 0.8571428571, 1, 0.9090909091, 1, 0.8333333333, 0, 0, 0, 0),
(13, 'Cerulean_Yes-Starship_Trooper:_A._Life_Seeker,_B._Disillu.mp3', 0.9444444444, 0.8947368421, 1, 0.9375000000, 1, 0.8823529412, 0, 0, 0, 0),
(14, 'Epiphyte_0122_heardemall.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(15, 'Epiphyte_0215_pointofknowreturn.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(16, 'Epiphyte_0228_ridethelightning.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(17, 'Epiphyte_0268_supersonic.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(18, 'Isophonics_03_-_Baby''s_In_Black.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(19, 'Isophonics_04_-_Erbauliche_Gedanken_Eines_Tobackrauchers.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(20, 'Isophonics_1-05 Rockin'' Robin.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(21, 'Isophonics_11_-_Ich_Kann_Heute_Nicht.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(22, 'Isophonics_12_-_I''ve_Just_Seen_a_Face.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(23, 'Isophonics_14_-_Santa_Donna_Lucia_Mobile.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(24, 'SALAMI_1448.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(25, 'SALAMI_1456.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(26, 'SALAMI_1504.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(27, 'SALAMI_1548.mp3', 0.9285714286, 0.8666666667, 1, 0.9166666667, 1, 0.8461538462, 0, 0, 0, 0),
(28, 'SALAMI_258.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(29, 'SALAMI_296.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(30, 'SALAMI_306.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(31, 'SALAMI_312.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(32, 'SALAMI_330.mp3', 0.9090909091, 0.8333333333, 1, 0.8888888889, 1, 0.8000000000, 0, 0, 0, 0),
(33, 'SALAMI_364.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(34, 'SALAMI_376.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(35, 'SALAMI_380.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(36, 'SALAMI_382.mp3', 0.9166666667, 0.8461538462, 1, 0.9000000000, 1, 0.8181818182, 0, 0, 0, 0),
(37, 'SALAMI_540.mp3', 0.9090909091, 0.8333333333, 1, 0.8888888889, 1, 0.8000000000, 0, 0, 0, 0),
(38, 'SALAMI_554.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(39, 'SALAMI_556.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(40, 'SALAMI_558.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(41, 'SALAMI_652.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(42, 'SALAMI_656.mp3', 0.9090909091, 0.8333333333, 1, 0.8888888889, 1, 0.8000000000, 0, 0, 0, 0),
(43, 'SALAMI_746.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0),
(44, 'SALAMI_786.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(45, 'SALAMI_8.mp3', 0.9090909091, 0.8333333333, 1, 0.8888888889, 1, 0.8000000000, 0, 0, 0, 0),
(46, 'SALAMI_818.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(47, 'SALAMI_888.mp3', 0.9000000000, 0.8181818182, 1, 0.8750000000, 1, 0.7777777778, 0, 0, 0, 0),
(48, 'SALAMI_898.mp3', 0.9285714286, 0.8666666667, 1, 0.9166666667, 1, 0.8461538462, 0, 0, 0, 0),
(49, 'SALAMI_904.mp3', 0.8888888889, 0.8000000000, 1, 0.8571428571, 1, 0.7500000000, 0, 0, 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `results`
--

CREATE TABLE `results` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `excerptID` int(10) NOT NULL,
  `version` int(10) NOT NULL,
  `rating` int(10) NOT NULL,
  `nWrongs` int(10) NOT NULL,
  `subjectID` int(10) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=0 ;

-- --------------------------------------------------------

--
-- Table structure for table `subjects`
--

CREATE TABLE `subjects` (
  `id` int(8) unsigned NOT NULL AUTO_INCREMENT,
  `first_name` varchar(512) DEFAULT NULL,
  `last_name` varchar(256) DEFAULT NULL,
  `email` varchar(256) DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `music_training` smallint(16) NOT NULL,
  `comments` varchar(4096) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=0 ;

