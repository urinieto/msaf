<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!--

Oriol Nieto
oriol -at- nyu -dot- edu
MARL, NYU

-->
<?php

    session_start();

    if (isset($_SESSION["results"]))
        $out = "set";
    else
        $out = "not set";
    echo $out . " ok<br/>";

    // Create connection
    $db_server = "localhost";
    $db_user = "urinieto_wp";
    $db_pass = "carambola1234";
    $db_name = "urinieto_boundaries_experiment2";
    $con = mysqli_connect($db_server, $db_user, $db_pass, $db_name);

    // Check connection
    if (mysqli_connect_errno($con)) {
        echo "Failed to connect to MySQL: " . mysqli_connect_error();   
    }

    // Get the excerpts that have the least amount of results
    $result = mysqli_query($con, "SELECT * FROM excerpts ORDER BY totalResults ASC");
    $least_results = mysqli_fetch_array($result)['totalResults'];
    $result = mysqli_query($con, "SELECT * FROM excerpts WHERE totalResults = $least_results ORDER BY RAND()");
    
    $excerpt = mysqli_fetch_array($result);

    // Find the version that has the least amount of results for the given excerpt
    $version_results = array(
        "v1res" => $excerpt['v1Results'],
        "v2res" => $excerpt['v2Results'],
        "v3res" => $excerpt['v3Results']
    );
    $version = array_keys($version_results, min($version_results))[0][1];
    //echo "Version: " . $version . "<br/>";
    //echo $excerpt['id'] . " ok<br/>";

    mysqli_close($con);

    // Set excerpt number
    $k = $excerpt['id'];

    // Set audio
    $audio = "audio/" . $k . "_v" . $version . ".mp3";
    //echo $audio . "<br/>";
    // TODO: Set name for storing results
    $name = "ratings";

    // TODO: Set the name of the variable to pass to the next page
    $rating_name = "score_rating_" . $k;
    $wrong_bounds_name = "score_bounds_" . $k;

?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="content-type" content="text/html; charset=utf-8" />
<title>::: Section Boundaries Experiment :::</title>
<meta name="keywords" content="" />
<meta name="description" content="" />
<link href="mfbasic.css" rel="stylesheet" type="text/css" />
<script src="audiojs/audiojs/audio.min.js"></script>

<script>

  var wrong_bounds = 0;

  audiojs.events.ready(function() {
    var as = audiojs.createAll();
  });

  function validateRadioButtons(ident) {
    var r1 = document.getElementsByName(ident)[0].checked;
    var r2 = document.getElementsByName(ident)[1].checked;
    var r3 = document.getElementsByName(ident)[2].checked;
    var r4 = document.getElementsByName(ident)[3].checked;
    var r5 = document.getElementsByName(ident)[4].checked;
    if (!r1 && !r2 && !r3 && !r4 && !r5) {
      return false;
    }
    else {
      return true;
    }
  }

  function validateForm() {
    if (!validateRadioButtons("ratings")) {
      alert("You must rate the boundaries of the track!");
      return false;
    }
  }

  function update_wrong_bounds() {
    document.getElementById("wbounds_text").value = wrong_bounds;
  }

  function add_wrong_bound() {
      wrong_bounds++;
      update_wrong_bounds();
  }

  function reset_wrong_bounds() {
      wrong_bounds = 0;
      update_wrong_bounds();
  }

</script>

</head>
<body>

<div id="top"></div>
<div id="container">
<div id="content">
<img src="images/MARL_cognition_header.jpg" width="780" height="71" alt="logo"/>
<h1>Section Boundaries Experiment</h1>

<p>The following excerpts have been marked with a "bell" sound for each section 
  boundary. Please, listen to them carefully and press the "Wrong Boundary!" button  every time you think that there is a boundary that shouldn't be there (false 
  positive) or when there is a boundary that should be there (false negative). 
  Finally, rate the overall quality of the boundaries based on your own judgement.
</p>

<p>
<form name="experimentform" method="post" action="index.php" onsubmit="return validateForm()">
<table width="760px">
    <?php
        echo '
        <tr>
            <td valign="top">
                <label for="s1_name_text"><font size="5"><strong>Excerpt '.$k.'</strong></font></label>
                <input type="hidden" name="'.$var_name.'" value="'.$name.'">
            </td>
            <td>
                Rating (1: Not Accurate, 5: Very Accurate)
            </td>
        </tr>
        <tr>
            <td valign="top">
                <center><audio src="'. $audio. '" preload="auto" /></center>
            </td>
            <td valign="center">
                1<input type="radio" name="'. $name .'" id="'. $name .'_1" value=1>&nbsp&nbsp&nbsp&nbsp
                2<input type="radio" name="'. $name .'" id="'. $name .'_2" value=2>&nbsp&nbsp&nbsp&nbsp
                3<input type="radio" name="'. $name .'" id="'. $name .'_3" value=3>&nbsp&nbsp&nbsp&nbsp
                4<input type="radio" name="'. $name .'" id="'. $name .'_4" value=4>&nbsp&nbsp&nbsp&nbsp
                5<input type="radio" name="'. $name .'" id="'. $name .'_5" value=5>&nbsp&nbsp&nbsp&nbsp
            </td>
        </tr>
        <tr>
            <td valign="top">
                <center><button type="button" id="wbound" onclick="add_wrong_bound();">Wrong Boundary!</button></center>
            </td>
            <td valign="center">
                <input type="text" id="wbounds_text" size=3 value="0" readonly>
                <button type="button" onclick="reset_wrong_bounds();">Reset</button>
            </td>
        </tr>';
  ?>
    
    <tr>
        <td colspan="2" style="text-align:center">
            <input type="submit" value="Submit Results">
        </td>
    </tr>
    </table>
</form>
</p>
</div>
</div>

<div id="footer">
    <a href="https://files.nyu.edu/onc202/public/" target="_blank">Oriol Nieto</a>, 
        Music and Audio Research Lab, New York University, 2014.
</div>

</div>
</body>
</html>
