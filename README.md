# **How to make a Simple, Android Pneumonia Detection App with SashiDo, Python and Kotlin**


##### Soumallya Dev


# Overview

With the recent Pandemic, the need of ML in the medical field is necessary more than year. Everyone nowadays is interested in ML, AI and Android, so this tutorial will aim to cover a little of everything.

We will be building an ML Android App that can detect pneumonia using X-Rays and can further detect if the pneumonia is Viral or Bacterial(If Detected)

We&#39;ll also be integrating our app with Sashido for push notifications for our users.

# Goals

- Make a predictive model that can detect Viral &amp; Bacteria Pneumonia
- Set up Android App
- Integrate the ML Model
- Connect SashiDo for Push Notifications

#

# Train our ML Model
### - Clone the GitHub Repo

  - Go to the https://github.com/soumallyadev07/Xanalyse and clone the the repository

### - Unzip TFLClassify.rar

### - Change Directory

  - Change your directory and enter into the folder called - &quot;Model Training&quot;
  ```bash
  cd Model Training
  ```
### - Go to https://www.kaggle.com/soumallyadev/pneumonia-xray-detection

  - Download the Given Data Set
  
### - Unzip chest_xay.zip

### - Open ModelTrain.py or ModelTrain.ipynb file

  - Open the above mentioned files, if you have Jupyter Notebooks configured, you can open the .ipynb file otherwise, you can simply open the .py file in your text editor
### - Understand the File Structure

  - You can find our training photos/data under the folder called - chest\_xray with 3 subfolders. These 3 folders are our categories and labels. If you want to add any more data, do remember to follow the same structure.
### - Install all necessary Dependencies

 - Our code consists of a lot of dependencies and try to install all of them before running to avoid the &quot;Module Not Found&quot; error and even if you get one, since we are learning, you can always search the error on Google and get 1000s of resources to solve them and solving errors is a crucial part of development.
 ```bash
  pip install MODULE_NAME
  ```
### - Run the Code &amp; Export the Model

  - Run your python program or each cell of the Jupyter Notebook and export the model in the same directory.
### - Rename your Model

  - Rename your model to &quot;XrayModel.tflite&quot;

# Set up the Android skeleton app

### Install Android Studio 4.1 beta 1

If you don&#39;t have it installed already, go [install AndroidStudio 4.1 Beta 1 or above](https://developer.android.com/studio/preview).

### Open the project with Android Studio

Open a project with Android Studio by taking the following steps:

- Open Android Studio After it loads select &quot;Open an Existing project&quot; from this popup:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/f3b8bea7e3b39376.png)

- In the file selector, choose TFLClassify/build.gradle from your working directory.

The project contains two modules, **start** and **finish**. If you get stuck, refer to the **finish** module for a hint.

- You will get a &quot;Gradle Sync&quot; popup, the first time you open the project, asking about using gradle wrapper. Click &quot;OK&quot;.

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/d68b4d7189e6c1e4.png)

- Enable developer model and USB Debugging on your phone if you have not already. This is a one-time set up. Follow [these instructions](https://developer.android.com/studio/debug/dev-options.html#enable).
- Once both your project and your phone is ready, you can run it on a real device by selecting TFL\_Classify.start and press the run button on the toolbar:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/60a77ef126c1373d.png)

- Now allow the Tensorflow Demo to access your camera:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/b63cba02bb36b7e3.png)

- You will see the following screen on your phone with random numbers taking the place of where real results will be displayed.

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/82c603596afa35f1.png)

### Add TensorFlow Lite to the Android app

- Select the start module in the project explorer on the left hand side:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/cede7f2b8b23c1a7.png)

- Right-click on the start module or click on File, then New -> Other -> TensorFlow Lite Model

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/bf243d9fdd27e20a.png)

- Select the model location where you have downloaded the custom trained XrayModel.tflite earlier.

Note that the tooling will configure the module&#39;s dependency on your behalf with ML Model binding and all dependencies automatically inserted into your Android module&#39;s **build.gradle** file.

![](https://i.ibb.co/517VbQj/Picture8.png)

- Click Finish.
- You will see the following at the end. The XrayModel.tflite is successfully imported and it shows the high level information regarding the model including the input / output as well as some sample code to get you started.

![](https://i.ibb.co/F43mWN6/Picture9.png)

### Checking out all todo list

TODO list makes it easy to navigate to the exact location where you need to update the codelab. You can also use it in your Android project to remind yourself of future work. You can add todo items using code comments and type the keyword TODO. To access the list of TODOs, you can:

- A great way to see what we are going to do is to check out the TODO list. To do that, select from the top menu bar View \&gt; Tool Windows \&gt; TODO

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/5de29b413574f25c.png)

- By default, it lists all TODOs in all modules which makes it a little confusing. We can sort out only the start TODOs by clicking on the group by button on the side of the TODO panel and choose Modules

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/5d8fe7b102340208.png)

- Expand all the items under the start modules:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/8d0f14a039995b20.png)

### Run the custom model with TensorFlow Lite

- Click on TODO 1 in the TODO list or open the MainActivity.kt file and locate TODO 1, initialize the model by adding this line:

```kotlin

privateclassImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :

ImageAnalysis.Analyzer-

...

// TODO 1: Add class variable TensorFlow Lite Model

private val xrayModel = XrayModel.newInstance(ctx)

...

}
```

- Inside the analyze method for the CameraX Analyzer, we need to convert the camera input ImageProxy into a Bitmap and create a TensorImage object for the inference process.

Note that current tooling requires image input to be in **Bitmap** format.

- This means if the input is a file rather than **ImageProxy** , you can feed it directly to the method **fromBitmap** as a **Bitmap** object.
- If you are interested in how **ImageProxy** is converted into **Bitmap** , please check out the method **toBitmap** and the **YuvToRgbConverter**. We expect the needs for these methods to be temporary as the team works towards **ImageProxy** support for **TensorImage**.

```kotlin
override fun analyze(imageProxy: ImageProxy) {

...

// TODO 2: Convert Image to Bitmap then to TensorImage

val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))

...

}
```

- Process the image and perform the following operations on the result:

- Descendingly sort the results by probability under the attribute score with the highest probability first.
- Take the top k results as defined by the constant MAX\_RESULT\_DISPLAY. You can optionally vary the value of this variable to get more or less results.

```kotlin
override fun analyze(imageProxy: ImageProxy) {

...

// TODO 3: Process the image using the trained model, sort and pick out the top results

val outputs = xrayModel.process(tfImage)

.probabilityAsCategoryList.apply {

sortByDescending { it.score } // Sort with highest confidence first

}.take(MAX\_RESULT\_DISPLAY) // take the top results

...

}
```

- Convert the sorted and filtered results into data objects Recognition ready to be consumed by RecyclerView via [Data Binding](https://developer.android.com/topic/libraries/data-binding):

```kotlin
override fun analyze(imageProxy: ImageProxy) {

...

// TODO 4: Converting the top probability items into a list of recognitions

for (output in outputs) {

items.add(Recognition(output.label, output.score))

}

...

}
```

- Comment out or delete the following lines which help generate the fake results we see before:

```kotlin
// START - Placeholder code at the start of the codelab. Comment this block of code out.

for (i in0..MAX\_RESULT\_DISPLAY-1){

items.add(Recognition(&quot;Fake label $i&quot;, Random.nextFloat()))

}

// END - Placeholder code at the start of the codelab. Comment this block of code out.
```
- Run the app on a real device by selecting TFL\_Classify.start and press the run button on the toolbar:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/60a77ef126c1373d.png)

- You will see the following screen on your phone with random numbers taking the place of where real results will be displayed.

## FINAL APP RUN THROUGH
![](https://s2.gifyu.com/images/ezgif.com-gif-maker-37e776b03bb7cb546.gif)

### Accelerate inference with GPU delegate

TensorFlow Lite supports several hardware accelerators to speed up inference on your mobile device. [GPU](https://www.tensorflow.org/lite/performance/gpu) is one of the accelerators that TensorFlow Lite can leverage through a delegate mechanism and it is fairly easy to use.

- Open build.gradle under the start module or you can click on TODO 5 under the TODO list and add the following dependency:

```kotlin
// TODO 5: Optional GPU Delegates

implementation "org.tensorflow:tensorflow-lite-gpu:2.2.0"
```
Note: we are adding this import manually in this codelab but when you do this in your project, you can just tick on the second box in the import screen to add GPU acceleration dependency

- Go back to the MainActivity.kt file or click on TODO 6 in the TODO list and initialize the following model option:

```kotlin
privateclassImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :

ImageAnalysis.Analyzer {

...

// TODO 6. Optional GPU acceleration

private val options = Model.Options.Builder().setDevice(Model.Device.GPU).build()

...

}
```

Note: there are multiple **Model** objects, choose the object **org.tensorflow.lite.support.model.Model** in your import

- Change the model initializer to use this by adding options to the method input:

```kotlin
privateclassImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :

ImageAnalysis.Analyzer {

...

// TODO 1: Add class variable TensorFlow Lite Model

private val flowerModel = xrayModel.newInstance(ctx, options)

...

}
```

- Run the app on a real device by selecting TFL\_Classify.start and press the run button ![](RackMultipart20210112-4-3pvk5h_html_afea2a9642ffaa7.png) on the toolbar:

![](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/img/60a77ef126c1373d.png)

On a medium/high end mobile device, GPU is much faster than CPU. Low end devices tend to have slower GPUs, so the speedup you see will vary.

### Connect SashiDo to Android Studio

 - Now go back to Android Studio and open your &quot;build.gradle(Project:Your project name)&quot; file
  - Add this code after the &quot;dependencies&quot; tag

```Kotlin
allprojects {
   repositories {
       maven{ url "https://jitpack.io"}
       google()
       jcenter()
   }
}
```

Now, go to your &quot;build.gradle(Module:app)&quot; file and add this implementation
```Kotlin
implementation "com.github.parse-community.Parse-SDK-Android:parse:latest_Jitpack_Version_Here"
```
Note: You can find the latest version at jitpack.io

![](https://res.cloudinary.com/practicaldev/image/fetch/s--amJD8HES--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/i/i3gw1cz1dpihw350hhc1.PNG)

Search this up in the Git repo URL box and press &quot;Look Up&quot; to get all the versions of Parse available.

Click &#39;Sync Now&#39; on the top section of your Android Studio Gradle file to sync your gradle files.

### Connect App using Parse

To make your app have access to the internet, go to your AndroidManifest.xml file (app>manifests>AndroidManifest.xml) and add this code before the application tag.

```Kotlin
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.INTERNET" />
```

Add this code in the &quot;application&quot; section in the AndroidManifest.xml file
```Kotlin
<meta-data
   android:name="com.parse.SERVER_URL"
   android:value="@string/SashiDo_server_url" />
<meta-data
   android:name="com.parse.APPLICATION_ID"
   android:value="@string/SashiDo_app_id" />
<meta-data
   android:name="com.parse.CLIENT_KEY"
   android:value="@string/SashiDo_client_key" />
```

Now go to the strings.xml file (app>res>values>strings.xml) and add this code
```kotlin
<string name="SashiDo_server_url">https://pg-app-tcbt8h75fl11zv9cv0zqzes6ebsjef.scalabl.cloud/1//</string>
<string name="SashiDo_app_id">[your app id here]</string>
<string name="SashiDo_client_key">[your client key here]</string>
```
Copy-paste your app id and client key from SashiDo. The server URL is under the API URL address box in SashiDo.
![](https://i.ibb.co/ys0h8vm/Screenshot-345.png)
Import the following:
```kotlin
import com.parse.Parse;
```
The class name should look like this:
```kotlin
public class App extends Application {
```
Now inside the class, inside the onCreate() function, add this code:
```kotlin
import android.app.Application;
import com.parse.Parse;

public class App extends Application {
   public void onCreate(){
       super.onCreate();
       Parse.initialize(new Parse.Configuration.Builder(this)
               .applicationId(getString(R.string.SashiDo_app_id))
               .clientKey(getString(R.string.SashiDo_client_key))
               .server(getString(R.string.SashiDo_server_url))
               .build()
       );
          }
}
```
Now let&#39;s test! In the main activity of your app, add this code:
```kotlin
ParseInstallation.getCurrentInstallation().saveInBackground();
```
Click run, and go to your SashiDo database. You should see the installation in your database.
![](https://i.ibb.co/0nDhmbB/Screenshot-346.png)
### Push Notifications

First, go to the Firebase console (make an account if you don&#39;t have one) and click &quot;Add Project.&quot; Enter a name for your project and press continue. On the Google Analytics page use the default account for firebase.

Once your project is created, click the android logo, and follow the instructions. You can find the package name of your app at the top of your AndroidManifest.xml file. (Should start with &quot;com.&quot;

![](https://i.ibb.co/bKwjDGn/Screenshot-348.png)
####
![](https://i.ibb.co/PZJWWjZ/Screenshot-350.png)

Then remember to put the google.json file in the app file of your app.

Finally, add this in your &quot;build.gradle(Project:Your project name)&quot; file in the dependencies tag:
```kotlin
classpath 'com.google.gms:google-services:4.3.3'
```
Add these implementations in the &quot;build.gradle(Module:app)&quot; file:
```kotlin
implementation "com.github.parse-community.Parse-SDK-Android:fcm:1.25.0"
implementation 'com.google.firebase:firebase-analytics:17.2.2'
implementation "com.github.parse-community.Parse-SDK-Android:fcm:1.19.0"
implementation 'com.google.firebase:firebase-core:17.2.2'
implementation 'com.google.firebase:firebase-messaging:17.2.2'
```
REMEMBER TO MAKE SURE YOU HAVE ALL THE IMPLEMENTATIONS LISTED ON THIS SCREEN!

Now, go to the project settings on firebase and then go to the cloud messaging tab and copy the Sender ID and the Server Key

Now, go to SashiDo&#39;s dashboard and go to &quot;App Settings&quot; and then &quot;Push&quot;. Then copy-paste the Sender ID and the Server Key into the appropriate boxes and click save.

To finish up, go to your AndroidManifest.xml file and add these 3 sections of code:
```kotlin
<service android:name="com.parse.fcm.ParseFirebaseMessagingService">
   <intent-filter>
       <action android:name="com.google.firebase.MESSAGING_EVENT" />
   </intent-filter>
</service>
```
```kotlin
<receiver
   android:name="com.parse.ParsePushBroadcastReceiver"
   android:exported="false">
   <intent-filter>
       <action android:name="com.parse.push.intent.RECEIVE" />
       <action android:name="com.parse.push.intent.DELETE" />
       <action android:name="com.parse.push.intent.OPEN" />
   </intent-filter>
</receiver>
```
```kotlin
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.VIBRATE" />
<uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />
<uses-permission android:name="android.permission.GET_ACCOUNTS" />
<uses-permission android:name="com.google.android.c2dm.permission.RECEIVE" />
```

To finish up, go to your App class, and add this to the onCreate() function.
```kotlin
ParseInstallation.getCurrentInstallation().save();
```
If you aren&#39;t able to see push notifications on the screen of your device, then try changing the above line of code to this:
![](https://i.ibb.co/PzxM3MV/Screenshot-351.png)

```kotlin
ParseInstallation installation = ParseInstallation.getCurrentInstallation()
installation.put("GCMSenderId", “<Your GCM SenderId”)
installation.saveInBackground()
```

Remember to put your Sender Id in the <> (found on the firebase console).

Now you can send push notifications! Run your app on a device/ emulator, and go to the SashiDo dashboard and click &quot;Push.&quot; Navigate to &quot;Send new push&quot; and type out your message. Make sure the preview is displayed on the android device, not the iPhone.
![](https://i.ibb.co/4ZWKnCG/Screenshot-352.png)
###
![](https://res.cloudinary.com/practicaldev/image/fetch/s--hCjVTm4x--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/i/9fjoswjz46vqnoh2fmgj.PNG)

Make sure the android button is selected

Finally, click &quot;Send&quot; and your device should get a notification!

![](https://i.ibb.co/CPt4x14/Whats-App-Image-2021-01-25-at-21-29-35-1-page-0001.jpg)

## References

-  [https://blog.sashido.io/tag/tutorial/](https://blog.sashido.io/tag/tutorial/)
    
-  [https://docs.parseplatform.org/android/guide/](https://docs.parseplatform.org/android/guide/)
    
-  https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/

-  https://blog.sashido.io/sashidos-getting-started-guide/#spreadyourmessagewithpushnotifications

# Closing Remarks
**I hope this tutorial is helpful for beginners and helps induce the passion of coding, development, Machine Learning and above all, problem solving.**
