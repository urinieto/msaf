<?php
    /**
     * Custom function to execute a specific uery for the given connection.
     */
    function my_exec_query($con, $query) {
        if (!mysqli_query($con, $query)) {
            throw new Exception('Error: ' . mysqli_error($con));
        }
    }

    /**
     * Create a new mySQL connection using the necessary credentials.
     */
    function create_connection() {
        $db_server = "localhost";
        $db_user = "urinieto_wp";
        $db_pass = "carambola1234";
        $db_name = "urinieto_boundaries_experiment2";
        $con = mysqli_connect($db_server, $db_user, $db_pass, $db_name);

        // Check connection
        if (mysqli_connect_errno($con)) {
            die("Failed to connect to MySQL: " . mysqli_connect_error());
        }

        return $con;
    }

    /**
    * Creates a new entry into the subjects table and returns the index
    * of the new entry in the database.
    */
    function create_new_subject($con) {
        $query = "INSERT INTO subjects (`first_name`) VALUES (NULL)";
        my_exec_query($con, $query);
        return mysqli_insert_id($con);
    }
    
    /**
     * Adds results to database with the given subjectID and connection.
     * It also updates the excerpts table with the new number of results.
     */
    function insert_result($con, $excerptID, $version, $rating, $nWrong, $subjectID) {
        $query = "INSERT INTO results (excerptID, version, rating, nWrongs, 
            subjectID) VALUES ($excerptID, $version, $rating, $nWrong, $subjectID)";
        my_exec_query($con, $query);

        // Get excerpt from excerpts table
        $result = mysqli_query($con, "SELECT * FROM excerpts 
            WHERE excerptID = $excerptID");
        $excerpt = mysqli_fetch_array($result);
        $ver_str = "v" . $version . "Results";

        // Increase version and total number of results
        $version = $excerpt[$ver_str] + 1;
        $totalResults = $excerpt["totalResults"] + 1;

        // Update database
        $query = "UPDATE excerpts SET $ver_str = $version, totalResults = $totalResults 
            WHERE id = $excerptID";
        my_exec_query($con, $query);
    } 

    /**
     * Update subject info to database with the correct subjectID.
     */
    function update_subject($con, $first_name, $last_name, $email, 
                            $music_training, $comments, $subjectID) {
        $query = "UPDATE subjects SET first_name='$first_name',last_name='$last_name',
            email='$email',music_training=$music_training,comments='$comments'
            WHERE id = $subjectID";
        my_exec_query($con, $query);
    } 

    /** 
     * Gets the next excerpt based on the one that has the least amount of results.
     * If there are more than one with the same amount of results, randomly pick one
     * of them.
     */
    function get_next_excerpt($con) {
        // Get the excerpts that have the least amount of results
        $result = mysqli_query($con, "SELECT * FROM excerpts ORDER BY totalResults ASC");
        $least_results = mysqli_fetch_array($result)['totalResults'];
        $result = mysqli_query($con, "SELECT * FROM excerpts 
            WHERE totalResults = $least_results ORDER BY RAND()");
        if (!$result) {
            die('Error: ' . mysqli_error($con));
        }
        return mysqli_fetch_array($result);
    }

    /**
     * Gets the version that has the least amount of results for the given excerpt.
     */
    function get_next_version($excerpt) {
        $version_results = array(
            "v1res" => $excerpt['v1Results'],
            "v2res" => $excerpt['v2Results'],
            "v3res" => $excerpt['v3Results']
        );
        // Sort the results, from low to high
        asort($version_results);

        // Return random results if they are all the same
        if ($version_results[0] == $version_results[1] &&
                $version_results[1] == $version_results[2]) {
            return rand(1, 3);
        }
        else if ($version_results[0] == $version_results[1]) {
            $rand_idx = rand(0, 1);
            return array_keys($version_results)[$rand_idx][1];
        }
        else {
            return array_keys($version_results)[0][1];
        }
    }

    /**
     * Reset database
     */
    function reset_database($con) {
        // Set number of results to zero
        $query = "UPDATE excerpts SET v1Results = 0, v2Results = 0, v3Results = 0, 
            totalResults = 0";
        my_exec_query($con, $query);

        // Delete all results
        $query = "DELETE FROM results";
        my_exec_query($con, $query);

        // Delete all results
        $query = "DELETE FROM subjects";
        my_exec_query($con, $query);
        
        // Close DB connection
        mysqli_close($con);
        $con = create_connection();

        // Reset auto increment indeces
        $query = "ALTER TABLE results AUTO_INCREMENT = 0";
        my_exec_query($con, $query);
        $query = "ALTER TABLE subjects AUTO_INCREMENT = 0";
        my_exec_query($con, $query);

        return $con;
    }

    /**
     * Sanitize strings in case there are hackers within your subjects
     */
    function sanitize_str($con, $str) {
        $str = strip_tags(addslashes($str));
        $city = mysqli_real_escape_string($con, $str); 
        return $str;
    }
?>
