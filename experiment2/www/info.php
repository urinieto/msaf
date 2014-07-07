<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!--

Oriol Nieto
oriol -at- nyu -dot- edu
MARL, NYU

-->
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="content-type" content="text/html; charset=utf-8" />
<title>::: Section Boundaries Experiment :::</title>
<meta name="keywords" content="" />
<meta name="description" content="" />
<link href="mfbasic.css" rel="stylesheet" type="text/css" />
</head>
<body>

<div id="top"></div>
<div id="container">
<div id="content">
<img src="images/MARL_cognition_header.jpg" width="780" height="71" alt="logo"/>
<h1>Section Boundaries Experiment</h1>

<p>Almost done. Please, fill in the following additional info.</p>

<p>
<form name="subjectform" method="post" action="submit.php">
<table width="760px">
    </tr>
        <tr>
            <td valign="top">
                <label for="first_name">First Name (optional)</label>
            </td>
        <td valign="top">
            <input  type="text" name="first_name" maxlength="50" size="30">
            <input type="hidden" name="subjectID" value="<?php echo $_POST["subjectID"]; ?>">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <label for="last_name">Last Name (optional)</label>
        </td>
        <td valign="top">
            <input  type="text" name="last_name" maxlength="50" size="30">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <label for="email">Email Address (optional)</label>
        </td>
        <td valign="top">
            <input  type="text" name="email" maxlength="80" size="30">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <label for="email">Years of Music Training</label>
        </td>
        <td valign="top">
            0<input type="radio" name="music_training" id="music_training0" value=0 checked>&nbsp&nbsp&nbsp&nbsp
            1<input type="radio" name="music_training" id="music_training1" value=1>&nbsp&nbsp&nbsp&nbsp
            2<input type="radio" name="music_training" id="music_training2" value=2>&nbsp&nbsp&nbsp&nbsp
            3<input type="radio" name="music_training" id="music_training3" value=3>&nbsp&nbsp&nbsp&nbsp
            4<input type="radio" name="music_training" id="music_training4" value=4>&nbsp&nbsp&nbsp&nbsp
            5<input type="radio" name="music_training" id="music_training5" value=5>&nbsp&nbsp&nbsp&nbsp
            6<input type="radio" name="music_training" id="music_training6" value=6>&nbsp&nbsp&nbsp&nbsp
            7<input type="radio" name="music_training" id="music_training7" value=7>&nbsp&nbsp&nbsp&nbsp
            8 or more<input type="radio" name="music_training" id="vocal_training8" value=8> 
        </td>
    </tr>
    <tr>
        <td valign="top">
          <label for="comments">Comments/Feedback</label>
        </td>
        <td valign="top">
          <textarea  name="comments" maxlength="1000" cols="25" rows="6"></textarea>
        </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:center">
            <input type="submit" value="Finish Experiment!">
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

<?php
  session_write_close();
?>
