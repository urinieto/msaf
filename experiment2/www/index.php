<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!--

Oriol Nieto
oriol -at- nyu -dot- edu
MARL, NYU

-->
<?php
    require 'utils.php';

    // Establish DB connection
    $con = create_connection();

    //$con = reset_database($con);

    try {
        // Get latest results (if exist)
        $excerptID = $_POST["excerptID"];
        $version = $_POST["version"];
        $ratings = $_POST["ratings"];
        $nWrongs = $_POST["nWrongs"];

        // Insert them to the database
        insert_result($con, $excerptID, $version, $ratings, $nWrongs, 
                    $_POST["subjectID"]);
        $_POST["i"]++; // New excerpt to evaluate
    }
    catch (Exception $e) {
        // First time to access the page
        // Get new subject ID and store it to the session
        $_POST["subjectID"] = create_new_subject($con);
        $_POST["i"] = 1;
    }

    // Get the correct excerpt ID and version
    $excerpt = get_next_excerpt($con);
    $version = get_next_version($excerpt);
    $excerptID = $excerpt['id'];

    // Close DB connection
    mysqli_close($con);

    // Set audio
    $audio = "audio/" . $excerptID . "_v" . $version . ".mp3";

    // Names for storing results
    $ratings_name = "ratings";
    $nWrongs_name = "nWrongs";
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
  var time_up = false;

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
    if (!time_up) {
      alert("You need more time to listen to the whole excerpt! Don't cheat you titan :-)");
      return false;
    }
    if (!validateRadioButtons("ratings")) {
      alert("You must rate the boundaries of the excerpt!");
      return false;
    }
    return true;
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

  setTimeout(function(){time_up = true;},90000);

</script>

</head>
<body>

<div id="top"></div>
<div id="container">
<div id="content">
<img src="images/MARL_cognition_header.jpg" width="780" height="71" alt="logo"/>
<h1>Section Boundaries Experiment</h1>

<?php
    $nExcerpts = $_POST["i"] - 1;
    $excerpts_str = ($nExcerpts == 1) ? "excerpt" : "excerpts";
    
    $instructions = "<p>
        The following excerpt has been marked with a \"bell\" sound for each section 
        boundary. A section boundary may occur when salient changes in various musical 
        aspects (such as harmony, timbre, texture, rhythm, or tempo) take place. 
        Please, listen to the excerpt carefully and:
        <ul>
        <li>Press the \"Wrong Boundary!\" button every time you think that there is a boundary that 
        shouldn't be there (false positive) or when you think there should be a boundary but there 
        was no bell sound (false negative). Note: The precise moment in time in which you press 
        the button is not relevant.</li>
        <li>Rate the overall quality of the boundaries based on your own judgement.</li>
        </ul><br/>
        </p>";
    $thank_you = "<p>Thanks! <strong>You have evaluated {$nExcerpts} {$excerpts_str}.
            </strong> You can evaluate as many as you like. 
            Whenever you want to finish the experiment, please press \"Finish\". ";
    $encourage = "But I encourage you to analyze at least 5 excerpts! :-)</p>";
    $reminder = "<p>As a reminder, here you have the intructions again: </p>";
    $finish_button = '<p><form name="submitform" method="post" action="info.php" 
        onsubmit="">
        <input type="hidden" name="subjectID" value="'.$_POST["subjectID"].'">
        <center><input type="submit" value="Finish"></form></center></p>';

    if ($_POST["i"] == 1) {
        echo $instructions;
    }
    else if ($_POST["i"] <= 5) {
        echo $thank_you;
        echo $encourage;
        echo $finish_button;
        echo $reminder;
        echo $instructions;
    }
    else {
        echo $thank_you . "<br/>";
        echo $finish_button;
        echo $reminder;
        echo $instructions;
    }
?>

<p>
<form name="experimentform" method="post" action="<?php echo $_SERVER['PHP_SELF'];?>" onsubmit="return validateForm();">
<table width="760px">
    <?php
        echo '
        <tr>
            <td valign="top">
            <label for="s1_name_text"><font size="5"><strong>Excerpt '.$_POST["i"].'</strong></font></label>
                <input type="hidden" name="excerptID" value="'.$excerptID.'">
                <input type="hidden" name="version" value="'.$version.'">
                <input type="hidden" name="i" value="'.$_POST["i"].'">
                <input type="hidden" name="subjectID" value="'.$_POST["subjectID"].'">
            </td>
            <td>
                Rating (1: Many mistakes, 5: No mistakes)
            </td>
        </tr>
        <tr>
            <td valign="top">
                <center><audio src="'. $audio. '" preload="auto" /></center>
            </td>
            <td valign="center">
                1<input type="radio" name="'. $ratings_name .'" id="'. $ratings_name .'_1" value=1>&nbsp&nbsp&nbsp&nbsp
                2<input type="radio" name="'. $ratings_name .'" id="'. $ratings_name .'_2" value=2>&nbsp&nbsp&nbsp&nbsp
                3<input type="radio" name="'. $ratings_name .'" id="'. $ratings_name .'_3" value=3>&nbsp&nbsp&nbsp&nbsp
                4<input type="radio" name="'. $ratings_name .'" id="'. $ratings_name .'_4" value=4>&nbsp&nbsp&nbsp&nbsp
                5<input type="radio" name="'. $ratings_name .'" id="'. $ratings_name .'_5" value=5>&nbsp&nbsp&nbsp&nbsp
            </td>
        </tr>
        <tr>
            <td valign="top">
                <center><button type="button" id="wbound" onclick="add_wrong_bound();">Wrong Boundary!</button>
                <input type="text" name="'. $nWrongs_name .'" id="wbounds_text" size=3 value="0" readonly></center>
            </td>
            <td valign="center">
                <button type="button" id="restart" onclick="reset_wrong_bounds();">Restart Counter</button>
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
        Music and Audio Research Lab, New York University, <?php echo date("Y"); ?>.
</div>

</div>
</body>
</html>
