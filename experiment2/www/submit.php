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

    // Sanitize strings before inserting into dataset
    $first_name = sanitize_str($con, $_POST['first_name']);
    $last_name = sanitize_str($con, $_POST['last_name']);
    $email = sanitize_str($con, $_POST['email']);
    $music_training = sanitize_str($con, $_POST['music_training']);
    $comments = sanitize_str($con, $_POST['comments']);

    // Update Subject
    update_subject($con, $first_name, $last_name, $email, $music_training, 
        $comments, $_POST['subjectID']);

    // Send email
    $email_message = "You have a new result!";
    $headers = 'From: '.$first_name."\r\n".
        'Reply-To: oriol@nyu.edu'."\r\n".
        'X-Mailer: PHP/' . phpversion();
    @mail("oriol@nyu.edu", "Boundaries Experiment", $email_message, $headers);  

    // Close DB connection
    mysqli_close($con);
?>

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
<h1>Thank you!</h1>

<p>The world is a bit better now :)</p>

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
    // Close connection
    mysqli_close($con);
    session_write_close();
?>
