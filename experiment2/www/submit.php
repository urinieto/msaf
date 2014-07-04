<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!--

Oriol Nieto
oriol -at- nyu -dot- edu
MARL, NYU

-->
<?php
    session_start();

    $_SESSION['first_name'] = $_POST['first_name'];
    $_SESSION['last_name'] = $_POST['last_name'];
    $_SESSION['email'] = $_POST['email'];
    $_SESSION['music_training'] = $_POST['music_training'];
    $_SESSION['comments'] = $_POST['comments'];

    // Create connection
    $db_server = "localhost";
    $db_user = "urinieto_wp";
    $db_pass = "carambola1234";
    $db_name = "urinieto_boundaries_experiment";
    $con = mysqli_connect($db_server, $db_user, $db_pass, $db_name);
    
    // Check connection
    if (mysqli_connect_errno($con)) {
      echo "Failed to connect to MySQL: " . mysqli_connect_error();   
    }    
       
    function died($error) {
          // your error code can go here
          echo "We are very sorry, but there were error(s) found with the form you submitted. ";
          echo "These errors appear below.<br /><br />";
          echo $error."<br /><br />";
          echo "Please go back and fix these errors.<br /><br />";
          die();
    }
    
    // Insert new row
    $sql = "INSERT INTO results (first_name, last_name, email, music_training,
        s1_name, s1_rating1, s1_rating2, s1_rating3,
        s2_name, s2_rating1, s2_rating2, s2_rating3,
        s3_name, s3_rating1, s3_rating2, s3_rating3,
        s4_name, s4_rating1, s4_rating2, s4_rating3,
        s5_name, s5_rating1, s5_rating2, s5_rating3,
        comments) VALUES
        ('$_SESSION[first_name]','$_SESSION[last_name]','$_SESSION[email]',
        '$_SESSION[music_training]',
        '$_SESSION[s1_name]', '$_SESSION[s1_rating1]','$_SESSION[s1_rating2]','$_SESSION[s1_rating3]',
        '$_SESSION[s2_name]', '$_SESSION[s2_rating1]','$_SESSION[s2_rating2]','$_SESSION[s2_rating3]',
        '$_SESSION[s3_name]', '$_SESSION[s3_rating1]','$_SESSION[s3_rating2]','$_SESSION[s3_rating3]',
        '$_SESSION[s4_name]', '$_SESSION[s4_rating1]','$_SESSION[s4_rating2]','$_SESSION[s4_rating3]',
        '$_SESSION[s5_name]', '$_SESSION[s5_rating1]','$_SESSION[s5_rating2]','$_SESSION[s5_rating3]',
        '$_SESSION[comments]')";

    //echo $sql;
    if (!mysqli_query($con,$sql)) {
        die('Error: ' . mysqli_error($con));
    }
    //echo "1 record added";

    // Update songs table
    /*
    $sql = "UPDATE songs SET n_results=n_results+1 WHERE 
        name='$_SESSION[s1_name]' OR name='$_SESSION[s2_name]'";
    if (!mysqli_query($con,$sql)) {
        die('Error: ' . mysqli_error($con));
    }
    */
    //echo "songs table updated $_POST[s1_name] $_POST[s2_name]";

    // Show results
    /*$result = mysqli_query($con, "SELECT * FROM results");
    while($row = mysqli_fetch_array($result)) {
        echo $row['s1_name'] . " " . $row['s2_name'] . "<br />";
    }
    */
    mysqli_free_result($result);
       
    // Send email
    $email_message = "You have a new result!";
    $headers = 'From: '.$_SESSION['firstname']."\r\n".
        'Reply-To: oriol@nyu.edu'."\r\n".
        'X-Mailer: PHP/' . phpversion();
    @mail("oriol@nyu.edu", "Boundaries Experiment", $email_message, $headers);  
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
</script>

</head>
<body>

<div id="top"></div>
<div id="container">
<div id="content">
<img src="images/MARL_cognition_header.jpg" width="780" height="71" alt="logo"/>
<h1>Thank you!</h1>

<p>The world is a bit better now :)</p>

</div>
</div>

<div id="footer">
    <a href="https://files.nyu.edu/onc202/public/" target="_blank">Oriol Nieto</a>, 
        Music and Audio Research Lab, New York University, 2013.
</div>

</div>
</body>
</html>
 
 
 
<?php
    // Close connection
    mysqli_close($con);
    session_write_close();
?>