<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!--

Oriol Nieto
oriol -at- nyu -dot- edu
MARL, NYU

-->
<?php

  /*
  // Create connection
  $db_server = "localhost";
  $db_user = "urinieto_wp";
  $db_pass = "carambola1234";
  $db_name = "boundaries_experiment";
  $con = mysqli_connect($db_server, $db_user, $db_pass, $db_name);
  
  // Check connection
  if (mysqli_connect_errno($con)) {
    echo "Failed to connect to MySQL: " . mysqli_connect_error();   
  }

  // Update songs table
  $result = mysqli_query($con, "SELECT * FROM songs ORDER BY n_results");
  //while($row = mysqli_fetch_array($result)) {
  //  echo $row['name'] . "<br />";
  //}

  $row_1 = rand(0,7);
  $row_2 = rand(0,7);
  while ($row_2 == $row_1) {
    $row_2 = rand(0,7);
  }

  //echo $result;
  if (!mysqli_data_seek($result, $row_1)) {
    echo "Cannot seek to row $row_1: " . mysql_error() . "\n";
  }
  $row = mysqli_fetch_assoc($result);
  $s1_name = $row['name'];
  if (!mysqli_data_seek($result, 0)) {
    echo "Cannot seek to row 0: " . mysql_error() . "\n";
  }
  if (!mysqli_data_seek($result, $row_2)) {
    echo "Cannot seek to row $row_2: " . mysql_error() . "\n";
  }
  $row = mysqli_fetch_assoc($result);
  $s2_name = $row['name'];
  mysqli_free_result($result);

  mysqli_close($con);
  */

    // TODO: Set excerpt number
    $k = 1;

    // TODO: Set audio
    $audio = "audio/0_v1.mp3";

    // TODO: Set name for storing results
    $name = "rating_" . $k;
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
    if (!validateRadioButtons("rating_1")) {
      alert("You must rate the boundaries of the track!");
      return false;
    }
  }
</script>

</head>
<body>

<div id="top"></div>
<div id="container">
<div id="content">
<img src="images/MARL_cognition_header.jpg" width="780" height="71" alt="logo"/>
<h1>Section Boundaries Experiment</h1>

<p>The following excerpts have been marked with a "ding" sound for each section 
  boundary. Please, listen to them carefully and rate the quality of the 
  boundaries based on your own judgement.</p>

<p>
<form name="experimentform" method="post" action="info.php" onsubmit="return validateForm()">
<table width="760px">
    <?php
        $var_name = "s". strval($i + 1) . "_name";
        echo '
        <tr>
            <td valign="top">
                <label for="s1_name_text"><font size="5"><strong>Excerpt '.$k.'</strong></font></label>
                <input type="hidden" name="'.$var_name.'" value="'.$name.'">
            </td>
            <td>
                Rating (1: Not Accurate, 5: Very Accurate)
            </td>
        </tr>';
        echo '
          <tr>
            <td valign="top">
                <audio src="'. $audio. '" preload="auto" />
            </td>
            <td valign="top">
                1<input type="radio" name="'. $name .'" id="'. $name .'_1" value=1>&nbsp&nbsp&nbsp&nbsp
                2<input type="radio" name="'. $name .'" id="'. $name .'_2" value=2>&nbsp&nbsp&nbsp&nbsp
                3<input type="radio" name="'. $name .'" id="'. $name .'_3" value=3>&nbsp&nbsp&nbsp&nbsp
                4<input type="radio" name="'. $name .'" id="'. $name .'_4" value=4>&nbsp&nbsp&nbsp&nbsp
                5<input type="radio" name="'. $name .'" id="'. $name .'_5" value=5>&nbsp&nbsp&nbsp&nbsp
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
