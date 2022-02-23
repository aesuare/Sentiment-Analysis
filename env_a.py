import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Read in and store in dataframe
messages_df = pd.read_csv('Datasets/Messages.csv')
scores_df = pd.read_csv('Datasets/Scores.csv')

# Create dictionaries containing message contents and scores
file_text = {}
for index, row in messages_df.iterrows():
    file_text[row['File Name']] = row['Text']
file_score = {}
for index, row in scores_df.iterrows():
    file_score[row['Merged']] = row['env_a']

sentences = []
scores = []


# The point of this for loop is to make sure that the file has both a score and an actual paragraph
for key in file_text:
    # I promise I did not want to have to do this but it seems I was left no other choice
    try:
        sentence = file_text.get(key)
        score = file_score.get(key)
        if sentence and score:
            sentences.append(sentence)
            int_score = float(score/5)
            scores.append(int_score)
    except:
        # print(f"Could not find key {key}")
        pass


# ===================================================================================================
# ===================================================================================================
"""
At this point, the only things that are good are the sentences and scores lists
They're in order, so if you zip iterrated the two you could see each letter and the subsequent score
"""
# ===================================================================================================
# ===================================================================================================


# Set model constants
EMBEDDING_DIM = 16
MAX_LENGTH = 300
NUM_EPOCHS = 30
TRUNC_TYPE = 'post'
OOV_TOK = '<OOV>'
VOCAB_SIZE = 50000


# Give each word an integer key
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index   # dictionary mapping each word to its integer counterpart

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=300, padding='post')

eighty_percent_mark = int(round(len(padded)*0.8))

# Divide lists into training testing
tr_pad = padded[:eighty_percent_mark]
tr_sco = scores[:eighty_percent_mark]
te_pad = padded[eighty_percent_mark:]
te_sco = scores[eighty_percent_mark:]


# Convert lists into numpy arrays
training_padded = np.array(tr_pad)
training_scores = np.array(tr_sco)
testing_padded = np.array(te_pad)
testing_scores = np.array(te_sco)



# Instantiate model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_padded, training_scores, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_scores))

model.summary()


test_sentences = ["""
I want to tell you about our exceptional performance in 2002 for our mission and business of expanding home- ownership in America. However, I also recognize the need for companies and their chief executive officers
to help restore and strengthen shareholder trust and confidence in corporate America. Certainly, Fannie Mae is no exception. Indeed, investor trust in our company is crucial to our business and mission as we raise capital from investors to finance homes.
Fannie Mae aspires to be a model company that inspires the confidence of our shareholders. As I describe our corporate performance for 2002, I also want to share with you Fannie Mae�s cutting-edge corporate governance practices that exemplify our values. It is our exceptional values that make our exceptional performance as a company possible.
Fannie Mae�s 2002 financial performance:
among our best years in history
Fannie Mae�s core business results for 2002 were among the best in the company�s history.
� Total business volume grew by 38 percent for a new record of nearly $850 billion.
� Total book of business grew by over 16 percent.
� Mortgage portfolio grew by nearly 12 percent.
� Outstanding mortgage-backed securities grew
by nearly 20 percent.
� Core taxable-equivalent revenue grew to $12 billion,
nearly 17 percent growth.
� Core business earnings per share (formerly, operating
earnings per share) grew by over 21 percent.
Fannie Mae produced this record year in spite of a weak economy and volatile markets for a very simple reason:
Our business is structured and managed to maintain disci- plined growth and to keep expanding homeownership in America throughout all economic conditions, both good and challenging.
What does �disciplined growth� mean at Fannie Mae?
For Fannie Mae, growth means that our core business earnings have grown faster than the Standard & Poor�s 500 index and the Nasdaq over the past 16 years.
For the past 16 years, Fannie Mae has produced double- digit increases in core business earnings per share, putting us among the best of the S&P 500 companies.
Fannie Mae�s strong, steady growth is based on the strong, steady growth of our market, which is the
stock of outstanding mortgages to be managed. That is why we say, �As the American Dream grows, so do we.�
During this decade, Fannie Mae�s market is projected to grow by 8-10 percent per year, which is faster than the 7 percent annual growth of the 1990s, and faster than most �growth� industries. Already our market
grew by over 10 percent in 2001 and over 12 percent in 2002. Moreover, Fannie Mae has grown faster than the market because we specialize in funding and managing the most popular, consumer friendly mortgage �
the long-term, fixed-rate mortgage.
Fannie Mae�s growth, however, is �disciplined,� meaning that we put a premium on stable financial performance and consistent return to investors under all economic conditions. The result has been extraordinarily low volatility and high stability in core business earnings growth for 16 years in a row. The fact that 2002 was an exceptional year for Fannie Mae in spite of the slow economy and unusually volatile interest rates is a testament to our disciplined growth model.
Letter to Shareholders
To Our Shareholders: In a challenging year for corporate America and the U.S. economy, the nation�s housing sector continued to be robust as the building, buying, and financing of homes produced another record year for the industry, especially for the chief source of funds for American families to buy homes, Fannie Mae.
FANNIE MAE 2002 ANNUAL REPORT 3

  4
FANNIE MAE 2002 ANNUAL REPORT
Fannie Mae produced this record year in spite of a weak economy and volatile markets for a very simple reason: Our business is structured and managed to maintain disciplined growth and to keep expanding homeownership in America throughout all economic conditions, both good and challenging.
Performance for Fannie Mae shareholders
is crucial to our mission
Fannie Mae�s performance for you, our shareholders, is crucial to our company. Your ownership makes possible our mission and business of expanding homeownership in this country, offering every family regardless of their means a better chance to achieve the American Dream. By growing our earnings and providing you with a good return, we can grow our capital. By growing our capital, we can grow our business. As Fannie Mae�s business grows, so do the benefits we provide � lower-cost mortgage funds � to more and more families in America.
For example, our disciplined business growth has steadily increased our core business earnings from 39 cents a share in 1987 to $6.31 in 2002, and we just increased the dividend on our common stock by 18 percent, which reflects our confidence in our earnings going forward.
At the same time, our disciplined growth also lowers conventional conforming mortgage rates. You can see this result in the mortgage rate charts published every Saturday in the real estate section of most newspapers. The mortgages Fannie Mae finances are always cheaper than the jumbo loans listed there, and much cheaper than subprime loans; we save home buyers anywhere from $11,000 to as much as $200,000 over the life of their loans. Furthermore, since 1987, the maximum loan amount eligible for Fannie Mae�s financing has more than doubled from around $150,000 to over $300,000, which means more homeowners can benefit from
our service.
Thus, Fannie Mae�s disciplined growth brings the
interests of shareholders and the interests of home buyers into perfect alignment. Our mission and our business complement each other. Your ownership of Fannie Mae makes more homeownership possible for more Americans.
Fannie Mae�s governance principles: openness, integrity, responsibility, accountability
What fuels the success of our mission and business are the principles that underly our approach to corporate governance and management: Openness. Integrity. Responsibility. Accountability. Fannie Mae puts a premium on upholding these simple, core principles in our corporate mission, business, and culture for an important reason: Trust is uniquely crucial to our company.
Fannie Mae�s mission is to raise private capital from investors in America and all over the world so that home buyers across our nation have a steady source of low-cost funds to finance homes. So we must earn and ensure the trust of investors, shareholders, and other stakeholders every day.
To earn and ensure that trust, Fannie Mae operates by the following principles of corporate governance:
Openness. Fannie Mae�s standard is to maintain best-in-class financial disclosures.
Our goal is to provide investors, shareholders, and other stakeholders with the clear, comprehensive information they need to understand and have confidence in Fannie Mae and make our financial disclosures easy to obtain and use. In our financial disclosures, we strive to provide more than is required, and anticipate and address fundamental questions about our company and business.
Our focus on transparency is not a new phenomenon for Fannie Mae.
In 2000, we revealed the 14 decision factors of our auto- mated mortgage underwriting system, Desktop Underwriter, so that lenders and borrowers could better understand our loan decision process and remedy any issues that arise with a loan application. We even started using our own credit assessment model so we could reveal those factors too.
Daniel H. Mudd
Vice Chairman and Chief Operating Officer

   Also in 2000, Fannie Mae adopted six voluntary initiatives to make us even more transparent and responsive to the market. We volunteered to:
1. Issue subordinated debt, which is a useful gauge of market confidence in the company;
2. Ensure we have at least three months� worth of liquidity in case our access to the public debt markets were to be disrupted;
3. Take a risk-based capital stress test every quarter and report the results;
4. Every quarter we give our books a credit risk shock test and report the results;
5. Every month we give our books three interest rate shock tests and report the results; and
6. We have obtained independent ratings of our financial strength and risk to the government, and those ratings are monitored continuously.
In 2002, we quickly adopted the leading new initiatives
to strengthen financial disclosure and accountability. Fannie Mae was one of the first companies to begin disclosing our insider stock trades in real time and to announce it would begin expensing stock options.
Also in 2002, Fannie Mae volunteered to register our common stock with the U.S. Securities and Exchange Commission and come under the SEC�s disclosure regime permanently. Fannie Mae�s financial reports now appear on the SEC�s EDGAR Web site.
In February 2003, Fannie Mae volunteered to add six new disclosures to the mortgage-backed securities we issue, providing investors with further details about the mortgages backing our securities.
In addition to providing cutting-edge disclosures, we try to make our financial disclosures easy to obtain and
easy to understand. We do not believe shareholders should have to dig through dense financial reports to get the information they need. And we don�t think they should have to puzzle out what our disclosures mean.
So we constantly are improving our Investor Relations Web site to make the information shareholders need easy to find and use. When we issue our monthly financial reports, we highlight what is most important about them.
We also believe it is important to give shareholders a straight answer to questions they might have, and
it is even better if the answer comes straight from the CEO. So Fannie Mae has launched a new section of
our corporate Web site called �Answers from the CEO,� where I personally address and answer the toughest questions we get.
Integrity. To ensure trust in Fannie Mae,
Fannie Mae must be trustworthy.
Every individual at Fannie Mae � from our Board of Directors to the Chairman and CEO to senior management to every employee � is held to the highest standards of honesty and integrity.
Integrity is woven into the Fannie Mae culture. As CEO, I strive to establish the highest standards of integrity by policy and by example. Our highly independent Board
of Directors holds me, as CEO, as well as senior management and the entire company, accountable for our high standards of integrity. The Board of Directors and each employee sign explicit Codes of Conduct, which
are available on our corporate Web site, fanniemae.com. Fannie Mae�s corporate justice system and a range of checks and balances provide three-dimensional protection of our integrity.
FANNIE MAE 2002 ANNUAL REPORT 5
 Jamie S. Gorelick
Vice Chair
 Timothy Howard
Executive Vice President and Chief Financial Officer

6
FANNIE MAE 2002 ANNUAL REPORT
Fannie Mae puts a premium on upholding our core principles of openness, integrity, responsibility, and accountability in our corporate mission, business, and culture for an important reason: Trust is uniquely crucial to our company.
Responsibility. The Chairman and CEO is responsible for Fannie Mae, its management and employees, and
to shareholders and other stakeholders.
As Chairman and CEO, I am personally responsible for ensuring that Fannie Mae operates in an effective, ethical manner that produces long-term value for shareholders.
I must not put my personal interests ahead of � or
in conflict with � the interest of the company or shareholders. Also, it is my duty to ensure that I know how Fannie Mae earns income and the risks we are undertaking in the course of business. Indeed, before it was required of us, Fannie Mae announced that our
CEO and Chief Financial Officer would sign and certify as to the honesty and accuracy of our financial statements, and we have a rigorous review process to ensure that.
As a CEO, one of the most offensive things about the corporate scandals that emerged recently was to hear CEOs claim that they did not know, they could not know, and they could not be expected to know about the activities that brought down their companies.
Accountability. Fannie Mae has a highly independent Board of Directors that selects and oversees the CEO. On behalf of shareholders, the Board of Directors
holds the CEO and senior management responsible for ensuring that Fannie Mae is operating effectively and with openness, integrity, responsibility, and accountability.
At Fannie Mae, the vast majority of our Board of Directors is independent from management. Board members are chosen for their experience, knowledge, and willingness to be active, engaged, and equipped to challenge the CEO and management on the competent and ethical operation of the company.
The Board and its Audit Committee ensure that management produces financial statements that are clear, complete, timely, and accurate. The Audit Committee, consisting of independent directors, has the sole authority to retain the independent auditor. The Audit Committee undertakes substantial due diligence to ensure that our auditors are independent, experienced, and well-qualified.
The Fannie Mae Board of Directors also includes a Compensation Committee and a Nominating and Corporate Governance Committee that are completely independent from management, which ensure that compensation and management choices and our corporate governance principles fully represent the interests of shareholders.
These core principles � openness, integrity, responsibility, and accountability � are the cornerstones of Fannie Mae�s corporate governance practices, which we make available on our new Corporate Governance Web site on fanniemae.com.
Standard & Poor�s praises Fannie Mae�s
corporate governance
To provide independent verification of our corporate governance and financial disclosures, Fannie Mae asked the independent ratings agency Standard & Poor�s
to examine and rate our standards. The report gave Fannie Mae an exceptional score of 9.0 on a 10-point scale, noting that, �Fannie Mae�s corporate governance practices are judged ... to be at a very strong level on a global basis of comparison.�
The Standard & Poor�s report stated, �in recent years, a combination of voluntary initiatives and specifics of [its regulatory] oversight have resulted in disclosure

Fannie Mae�s strong, stable, consistent performance for shareholders and homeowners is built on the character and conscience of our 4,700 employees and the Board of Directors that oversees and holds us accountable to you.
about Fannie Mae�s financial health that is unavailable from other, similar financial institutions.�
The report affirmed that the structure of our Board of Directors meets or exceeds the latest rules on board composition proposed by the New York Stock Exchange, and that our board has a clear and substantial majority
of independent, non-executive directors and the right independent board committees.
As the Standard & Poor�s report states, Fannie Mae�s Board of Directors �combines a good mix of new and longer-serving directors, directors of high caliber and with a diversity of skills and a strong voice of independence and engagement. ... [T]he board appears to be an effective leader of the company and monitor of management. ... [D]irectors appear to be engaged and show a desire
to demonstrate leadership in board effectiveness and governance. ... Fannie Mae�s audit committee demonstrates a commitment to the independence of the audit process. Its members are actively engaged with both the internal audit team and the outside auditors.�
Character and conscience
In its report on our corporate governance, Standard & Poor�s stated, �By being the first U.S. company to publish
its governance score from Standard & Poor�s,
Fannie Mae is not only demonstrating its own strong governance practices, but is also showing leadership
in the United States with regard to providing greater openness and disclosure about its corporate governance standards.�
Fannie Mae is proud of our leadership in corporate governance, but never satisfied with the status quo.
We also recognize that earning the trust and confidence
of shareholders and the public we serve requires more than cutting-edge corporate governance and financial disclosure policies. As the scholar Michael Novak has noted, corporate responsibility ultimately requires
two essential qualities: personal character and conscience.
Fannie Mae�s strong, stable, consistent performance for shareholders and homeowners is built on the character and conscience of our 4,700 employees and the Board of Directors that oversees and holds us accountable to you. Thank you for believing in us.
Franklin D. Raines
Chairman and Chief Executive Officer
 FANNIE MAE 2002 ANNUAL REPORT
7
""",
"""
Winn-Dixie Stores, Inc. 2000 Annual Report - Message to Our Shareholders

Message to Our Shareholders
For Winn-Dixie, fiscal 2000 was a year of pride, disappointment and, ultimately, renewed optimism. As our 75th year in business, it was a time to reflect on the Company's proud heritage, but also a time to chart a new path for the future. It was a disappointing year because Winn-Dixie continued to perform below acceptable levels, and our plan to sell the Texas and Oklahoma operations was opposed by the Federal Trade Commission. Nevertheless, despite these setbacks, Winn-Dixie initiated the most significant corporate restructuring effort in its history. Also, after reevaluation, we believe there is opportunity for growth in Texas and Oklahoma. Immediately after my appointment as President and CEO last November, our restructuring process began in earnest. At that time we initiated a thorough process of self-analysis at Winn-Dixie designed to identify best industry practices in order to address two primary issues of concern: First, how to do the best job possible of serving our customers' needs in each store. Secondly, how to coordinate and organize all of Winn-Dixie's operations so that they serve our customers most efficiently. The process evolved to touch upon more areas of the Company than originally anticipated. In doing so, it created an experience that was challenging and purposefully at odds with the old ways. But it also proved to be reinvigorating, and because of the energies it released, it has served to infuse Winn-Dixie with renewed optimism and confidence for the future.

2000 Annual Report Winn-Dixie Stores, Inc.

Dan Lafever, Senior Vice President & Director of Operations

Our self-examination began by identifying our weaknesses at the store level through the eyes of our customers: what did they think about our stores, our operations and our people? We came to understand that our stores need to be cleaner, better stocked and our service to the customer a whole lot friendlier and more helpful. In order to achieve these objectives, we must create a structure that will support a stronger store management team. Our Store Managers must have direct responsibility for meeting customers' needs and expectations. After all, they are the ones closest to our customers. So we have empowered them to lead their associates and to run their stores in the manner best suited for their individual markets and communities.

Al Rowland, President, Chief Executive Officer & Chairman of the Executive Committee

Rick McCook, Senior Vice President & Chief Financial Officer

To assist them in that effort, teams, led by a District Manager and consisting of experts in various product categories, have been established to support the needs of our Store Managers. Each team serves approximately 20 stores and helps to coordinate extensive training and service validation programs at all levels of store operations. Concurrent with our efforts at the operational store level, we have instituted a centralization program that has reassigned many responsibilities not directly linked to our retail operations to our

Brought to you by Global Reports http://www.winndixie.com/company/annualreport/2000/message.htm (1 of 3) [4/18/2001 9:32:10 PM]

Winn-Dixie Stores, Inc. 2000 Annual Report - Message to Our Shareholders

corporate headquarters in Jacksonville, Florida. Doing so will assure that the energies of our management personnel at the division level remain focused on our customer-centered objectives at store level.
John Sheehan, Senior Vice President & Director of Sales and Procurement

As part of the centralization effort, warehouse and fleet operations, new store development and construction programs and existing store maintenance are now directed out of corporate headquarters in Jacksonville. This will provide more consistency, better coordination of resources as well as considerable cost savings. To achieve more efficient operations overall, we conducted a comprehensive and aggressive analysis of all of our support operations and staff functions, which have resulted in the following changes: Management responsibilities for our Midwest, Atlanta, and Tampa divisions have been assumed by neighboring divisions. We closed 111 unprofitable stores and two manufacturing operations were discontinued. We combined our private label and manufacturing operations. We centralized our purchasing, merchandising and marketing functions for all divisions into one group located in Jacksonville. We have assigned 650 stores for remodeling in 2001 in order to improve their appeal and functionality.

August Toscano, Senior Vice President & Director of Human Resources

    
Ellis Zahra, Senior Vice President  & General Counsel

At corporate headquarters, the staff was reorganized, with significant emphasis placed on the direction of the Company's training and human resource functions. Going forward, our goal is to field the best associates, at all levels, in every store.  Taken together, these changes affected over 11,000 positions. While it was difficult to eliminate jobs, we implemented a severance package for those qualifying associates who could not assume other positions within our organization. The immediate impact of these developments is significant and, unfortunately, painful: Winn-Dixie has incurred a charge to earnings of $396 million in 2000 and we expect to incur additional charges of approximately $144 million in 2001. However, on the positive side of the ledger, we anticipate achieving a considerable longer term benefit as a direct result: a savings of approximately $400 million per year in expenses beginning about one year following completion of our restructuring program and implementation of improved store operations and support. Structural and operational improvements, combined with annual cost savings, will serve to enhance our competitive position and, in turn, positively impact our sales and profitability.

"From time to time, every business has to reevaluate the way they do things, and sometimes the old ways are no longer the best. Making tough decisions is the only way to move forward." -Al Rowland

Seventy-five years is a significant milestone for any business, and a company that achieves such an anniversary has undoubtedly weathered many challenges along the way. But to survive and prosper in the years ahead, every successful enterprise needs to embrace change as a necessary business process.

Brought to you by Global Reports http://www.winndixie.com/company/annualreport/2000/message.htm (2 of 3) [4/18/2001 9:32:10 PM]

Winn-Dixie Stores, Inc. 2000 Annual Report - Message to Our Shareholders

While 2000 was a disappointing year for Winn-Dixie, the organization has embraced change and is fully committed to proceed through that dynamic process. We have taken dramatic and necessary steps to reposition the Company for the future, and we will not waver in our goal of creating a new Winn-Dixie for the new century.

Al Rowland President and Chief Executive Officer

Brought to you by Global Reports http://www.winndixie.com/company/annualreport/2000/message.htm (3 of 3) [4/18/2001 9:32:10 PM]


""",
"""
USG Corporation 2000 Annual Report

Dear Fellow Shareholders

USG enjoys tremendous strengths, including leading brands, low-cost

production, significant liquidity and exceptional people. They are vital to our ability to weather the conditions we face today. In ordinary circumstances, we'd call 2000 a year of accomplishment. We posted the second highest sales in our history. Record shipments of our major product lines rolled from new, low-cost production facilities. We increased our share of our markets. We repurchased 5 .7 million shares of our stock. We improved our ability to serve a changing marketplace. But the circumstances are far from ordinary. Almost overnight, the wallboard industry swung from unmet demand to excess supply, just as energy and raw material costs started to climb. Wallboard prices that reached $166 per thousand square feet in December 1999 fell to
$94 per thousand square feet by December 2000 . Energy costs more

than doubled in the fourth quarter alone. At the same time, a wave of asbestos litigation rose to threaten the future of our company. To help manage the uncertainty surrounding the asbestos issue, we took a year-end charge of $850 million to increase reserves for settling asbestos claims filed through 2003 . We also recorded a $54 million charge for improving our efficiency and streamlining our operations. Together, these events kept us from achieving a sixth straight year of record earnings growth. While net sales of $3 .78 billion nearly equaled the record $3 . 81 billion we reported in 1999 , our net earnings declined dramatically. Excluding the special charges, net earnings fell to $298 million, or $6 . 49 per fully diluted share. After the charges, we recorded a loss for the year of $259 million, or $5 . 62 per fully diluted share. We are disappointed to report results like these, to say the least, although there is some consolation in the fact that, operationally, we continued to outperform others in the industry. But the key question now is what are we doing to respond to these challenging conditions? How will we regain our momentum? We'll adapt. We'll change. We'll perform.

6

Adapt

First, we must do everything we can to resolve the asbestos litigation crisis and manage our asbestos liability costs.
USG never mined, made or sold raw asbestos. But we did use it as a

minor ingredient -- typically less than five percent -- in some of our plasters and joint compounds. And we have long supported efforts to fairly compensate people whose health was affected by those products. It hasn't been easy. When it comes to dealing with asbestos litigation, the U.S. legal system has clearly broken down. Courts are inundated with asbestos exposure claims, the vast majority of which have been submitted on behalf of people who show no signs of impairment. The weight of this litigation has dragged more than two dozen companies into bankruptcy. We are fighting to avoid that fate. Yet each new bankruptcy leaves fewer companies to shoulder the burden. Our balance sheet and the loans that banks have committed to make us provide significant liquidity. But while we are equipped to manage our own liability, we cannot take on the responsibility for the entire industry. We agree with the Supreme Court that resolving the issue calls for national legislation, and we have joined efforts to craft a solution that will keep the asbestos litigation crisis from becoming a disaster. In the meantime, our $850 million pretax asbestos charge covers the costs we expect to incur for asbestos claims filed in the next three years. It does not close the door on the issue, but it will bring a higher degree of certainty to our financial performance as we move forward. We also must adapt to new conditions in our markets. Throughout our
99-year history, our goal has been to lead at every point in the economic

cycle -- trough-to-trough and peak-to-peak. And that is what we are doing today, by continuing to carry out our strategic plan.
w

WCF

WCF RHF JSM

William C . Foote, Chairman, CEO and President Richard H . Fleming, Executive Vice President and CFO James S . Metcalf, Senior Vice President; President and CEO , L&W Supply

7

Our leadership begins with the industry's most advanced production facilities. Over the past five years, we invested in excess of $1 billion, primarily to build new production facilities and improve our operations. It was the right strategy. We had to expand to meet our customers' needs or risk losing them as customers. And it was increasingly expensive to run our existing plants, some which dated back to the 1930 s. In 2000, we began to reap the benefits of our investments. With 3.3 billion square feet of new, low-cost wallboard production capacity, we are better able to remain profitable in a time of falling prices. Advanced production technologies also help cushion the impact of rising energy costs. Now, we're turning our attention to strengthening our cash flow. Anticipating continued softness in the wallboard market, we cut our capital expenditures by five percent during the second half of the year and reduced them even more as we entered 2001. Our $54 million restructuring plan closed three high-cost production lines and eliminated over 500 jobs, out of a total of 4 , 400 salaried positions. We have now closed six wallboard production lines since 1999 -- a total of approximately 1. 5 billion square feet of high-cost capacity, or almost
50 percent more than originally planned. We also have reduced our

quarterly dividend to shareholders. Actions like these are never easy, but together, they will strengthen our cash flow by approximately $50 million each year. We will devote even more attention to satisfying our customers. Because we are the preferred wallboard brand, we gain market share in times of free supply, and that is happening again. But we aren't taking anything for granted. Our customers' needs and expectations are changing, and we are changing along with them.

EMB JHM MSK

Edward M . Bosowski, Senior Vice President; President, International John H . Meister, Senior Vice President; President, Building Systems Marcia S . Kaminsky, Vice President, Communications

11

Change

One of the most significant changes is the convergence of our wallboard and ceilings customers. Close to three-quarters of our customers now sell both of our major product lines. We're positioning ourselves to provide them with comprehensive, convenient and cost effective service. New products and services are helping us meet the demand for greater value and improved performance. In January 2000 , we reinvented our largest product line with the introduction of U . S . Gypsum's "Next Generation" SHEETROCK brand gypsum panels that offer cleaner scoring and snapping, improved durability and faster installation. U . S . Gypsum also introduced a new family of FIBEROCK sheathing products that is rapidly winning a place in the market. USG Interiors' ceiling products, including CURVATURA ELITE curved grid, allow architects greater creativity and flexibility in their designs. Meanwhile, Design Wizard, a new one-of-a-kind web-based design tool, enables specifiers to create, engineer and print a bill of materials for new ceiling designs. Our work is unfinished. More challenges lie ahead. In 2001, we expect little, if any, growth in our markets. Although there are signs that lower interest rates may give the construction industry a shot in the arm, we will continue to face tough times until the supply of wallboard comes back into balance with demand. High energy costs and other inflationary pressures also are a concern.
Perform

Little is certain in such an environment, except our commitment to leading our markets -- in good times and bad. Our strategic plan provides a roadmap for remaining the low-cost producer, increasing our market share and improving our competitiveness. Now, it's time to perform. We'll focus on a limited number of key issues: Pushing for a fair, comprehensive solution to the asbestos litigation crisis. Quantifying our projected liability through 2003 does not solve the fundamental problem of how asbestos cases are handled in the tort system. Finding a solution that all parties can accept will not be easy, but the recent bankruptcies of otherwise healthy companies benefit no one. We will work hard to help shape and win approval for a national legislative solution.

14

Optimizing our new plants. In 2000 , we concentrated on completing our new production facilities. In 2001, we will focus on making the most of them. As the new facilities have come on-line, we have been fine-tuning the operations to achieve optimal efficiencies and savings. And because we produce the preferred product, we expect to run at higher utilization rates than our competitors. Managing energy costs, working capital and capital expenditures. We'll work harder than ever to manage our energy costs. In fact, as we entered 2001, we already had purchased more than two-thirds of the natural gas we will use during the year, to hedge against future increases. We'll continue to find new ways to improve the management of our inventory, investments and accounts receivable. We plan to reduce capital spending significantly during 2001. We will defer all non-essential capital projects and explore the sale of assets that are not vital to our business. Satisfying our customers. We are operating in a buyer's market. So we must earn our leadership position every day, by providing both excellent products and superb customer service. The changes we have made to the company provide the structure and strengths we need to meet our customers' evolving needs. Now, we must apply those advantages to build stronger relationships with our customers and help them grow and achieve their goals. We want our customers to value USG as much as we value our customers. Although the tests we face bring uncertainty and risk, passing through them will make us stronger and more flexible. And no company in our industry is better prepared to meet these challenges. We have the right strategies. Our brands lead their markets. We're the low-cost producer. Most of all, we have the people. In difficult times, the men and women who work at USG have once again proven that they are the best in the business -- always committed to the better way -- and they have earned my gratitude and respect. With their experience, creativity and plain hard work, we will continue to adapt, change and perform -- and return to the unfinished business of growth.
w

WCF

William C . Foote Chairman, CEO and President February 14, 2001

15


""",

"""
DEAR FELLOW STOCKHOLDERS,
Fiscal 2008 was a very successful year for Symantec, as demonstrated by our solid performance against key financial metrics. In addition to strong revenue and earnings growth, we also achieved significant deferred revenue and cash flow generation results.
We believe we are positioned for even stronger performance in fiscal year 2009 based on three key strategic pillars:
� A Clear Growth Path. Symantec�s strategy � to secure and manage our customers� infor- mation-driven world � positions us to leverage existing strengths into new opportunities, while also capitalizing on emerging industry growth needs.
� Delivering on the Promise of Innovation. Symantec is successfully innovating through a combination of organic development, strategic acquisitions, and strong technology partnerships.
� Day-In, Day-Out Execution. Symantec�s focus on execution is clearly demonstrated by our internal actions and our ongoing financial results.
2008 � A SOLID YEAR
Throughout fiscal year 2008, we made significant progress towards our goal of cross-selling and up-selling an expanded portfolio of Symantec�s products and services to both new and existing customers. In addition, important adjustments made in our sales and marketing programs, as well as improved execution, fueled more than 1,500 large deals during the year.
We fortified our position at the endpoint with the addition of Altiris and Vontu. The combination of Symantec and Altiris allows our customers to better manage and enforce security policies at the endpoint, identify and protect against threats, remediate vulnerabilities, and manage valuable IT assets. With Vontu, we give our customers the ability to determine what data they should protect and how they should protect it. These transactions represent a natural extension of our security strategy � that the most secure endpoint is a well managed endpoint.
We launched innovative new products in segments of the market where we already hold strong leadership positions. For example, we added disk based backup to our market leading data protection products and introduced enterprise storage management to our foundation platform, enabling more efficient use of storage resources. We also created a single software agent for addressing multiple security threats on the enterprise endpoint. Lastly, we introduced online backup to both consumers and enterprises. In total, we exited the fiscal year with the strongest product portfolio we�ve ever had.
A strong product portfolio, however, will deliver less value if we don�t focus on the fundamentals to successfully manage the business. The operational improvements made throughout the year should yield better bottom-line performance. I�m pleased to report that we were able to better manage our cost structure and delivered on our long-term objective of improving operating margins by 100 basis points per year. Symantec secures and manages the information-driven worldagainstmorerisksatmorepoints,morecompletelyandefficientlythananyothercompany.
1

FINANCIAL PERFORMANCE
In Fiscal 2008, we achieved both record revenue and earnings per share. Non-GAAP revenue1 grew 13% to more than $5.93 billion, generating non-GAAP earnings per share1 of $1.27. Non- GAAP deferred revenue1 grew 12% to nearly $3.1 billion and we generated cash flow from operating activities of $1.8 billion, up 9% compared to Fiscal 2007. Given the recurring nature of our business model, deferred revenue and cash flow from operations are important metrics in measuring the overall strength of our business.
During the year, we continued to demonstrate our commitment to creating shareholder value by repurchasing a total of $1.5 billion of our common stock.
WELL POSITIONED FOR FISCAL 2009
We entered the new fiscal year with a strong sales pipeline and we are well positioned for continued success. During this fiscal year we intend to leverage our core strengths in security, storage, and data protection to accelerate growth in high potential areas, up sell new functionalities and drive incremental business. Specifically, we plan to grow our core business franchises at or above market growth rates to continue fueling our cash flow, scale our high-growth businesses to contribute materially to our top-line revenue growth, and seed emerging growth to keep us relevant in the long run. In addition, we plan to exceed market growth rates in the fastest growing international markets and use mergers and acquisitions to complement our product portfolio growth.
Several areas of focus this fiscal year should further strengthen our operating returns:
� Weplananumberofkeyproductintroductionsduringtheyear,buildinguponourrichportfolio of products and services.
� We will leverage the new technologies and businesses we have successfully acquired across our portfolio.
� We are refocusing our spending toward higher growth areas while we continue to improve operating returns. We will capitalize on emerging industry growth trends such as data loss prevention, endpoint virtualization, Software-as-a-Service (SaaS) and consumer services.
Data Loss Prevention (DLP). Our acquisition of Vontu expanded Symantec�s presence in the rapidly growing data loss prevention market and provided us with clear market leadership and product functionality at all tiers: the network, storage and endpoint, all managed from a central console. To complement our DLP capability, Symantec is partnering with Guardian Edge to deliver proven endpoint encryption products. Both DLP and endpoint encryption are key com- ponents in helping customers protect valuable information that resides on laptops and desktops. We see excellent opportunities to broaden the distribution of our DLP and encryption products, particularly in international markets, and to integrate these capabilities with several of our key products including mail security and archiving.
Endpoint Virtualization. New technologies, like virtualization, are evolving to enable more efficient management and flexible use of servers and endpoints. Symantec�s application virtualization technology is changing the way software is managed, delivered and consumed
1 Non-GAAP results are reconciled to GAAP results on page 5. 2
  
at the endpoint. Over time, as endpoints evolve to incorporate a range of computing and delivery models � local or streamed applications within physical or virtual desktops � Symantec will provide solutions that support and manage this complexity. Our recent acquisition of AppStream, whose capabilities are already incorporated into our Software Virtualization Services products, builds on our endpoint management and virtualization portfolio.
Software-as-a-Service. In mid-February, we launched two services from the Symantec Protection Network, which is our new Software-as-a-Service (SaaS) business designed to provide small- and medium-sized customers with a suite of online data protection and security solutions. Symantec Online Backup provides data protection services for servers, desktops and laptops online, while Symantec Online Storage for Backup Exec provides a disaster recovery service for mid-sized customers. These solutions are designed to provide best-in-class protection for our customers� mission critical business data. We expect to deliver additional SaaS offerings in areas where we have market-leading products such as endpoint security, archiving and messaging.
Consumer Services. We will continue to expand our market leadership in the consumer business by introducing innovative products and services. We are driving incremental revenue per customer as they migrate from point products to suites and begin consuming value added services such as our expert installation, system checks and PC performance tune-up offering. Customer feedback has been strong and we believe that our consumer services will underpin our growth objective for the business this year.
We are additionally focused on meeting our stakeholder�s expectations for continued leadership in good governance; the greening of IT; and advocacy for privacy, data protection, and online safety. These priorities, along with other areas of corporate responsibility such as our work to promote employee diversity and our outreach to local communities, go beyond �nice to have� initiatives. Rather, they are intertwined with our core business objectives and can have a real and significant impact on financial performance and our company�s long-term success in the market.
We have taken several important steps this past year to advance our corporate responsibility performance. The Nominating and Governance Committee of our Board of Directors amended its charter to include oversight of corporate responsibility issues. We completed our first global greenhouse gas inventory in preparation for setting carbon dioxide reduction targets. We are using our products internally to reduce energy consumption and assisting our customers in reducing their own energy needs. We adopted the Calvert Women�s Principles and reiterated our support for the Ten Principles of the United Nations Global Compact. These actions, and many more, speak to our commitment to larger social and environmental issues. We are eager to continue this work and invite our stakeholders to partner with us.
CONFIDENCE IN OUR FUTURE
In closing, let me underscore my confidence in our business and our future. Symantec is a great company with superb brands and a very talented team. More than ever, our customers and partners look to Symantec to help them secure and manage their information across the full spectrum of information technology platforms. In the face of an ever changing technological landscape, we are confident that our business is well-positioned as we enter the new fiscal year. We look forward to delivering on those expectations and have no doubt we will succeed.
  3

On behalf of the Board, I sincerely thank our employees for another year of tremendous effort � and corresponding results. I also extend our gratitude to our partners and customers for their loyalty to Symantec. Finally, I thank our stockholders for their support as Symantec continues to grow and evolve in the global marketplace.
Sincerely,
JOHN W. THOMPSON
Chairman of the Board and Chief Executive Officer
  4
""",

"""
        Dear Fellow Stockholders:
Despite the turmoil in the global economy in 2009, we maintained a balanced approach of focusing on both short- and long-term goals. We were diligent in our execution and resolve in achieving our 2009 operational and financial goals. We also remained focused on our longer-term strategies for sustainable growth, which include:
� Staying focused on the customer;
� Innovation leadership with ongoing solution and service development in both core and new market
applications;
� Capitalizing on growth in developing regions;
� Focusing on cash flow and return on assets;
� Driving efficiencies and productivity in our business to maximize profitability;
� Sustainability in everything we do; and
� Developing our people.
During the year, we remained confident about our business model, our performance and our prospects. Our view is reinforced by our experience in emerging from economic downturns as a stronger organization, having a platform that is in place and optimized for more profitable and accelerated growth in developing regions, our solid cash flow generation, leading industry brands that are valued worldwide, extensive intellec- tual property, and a business driven by a diverse, talented and passionate team�16,200 strong.
This year�s annual report theme, �It�s not just what we make, it�s what we make happen,� succinctly describes our business model and how it sets us apart in the marketplace: our innovative leadership, the positive impact our solutions have made in the world, and in the measurable value we have created for customers.
It is especially fitting as we celebrate our 50th anniversary of incorporation and the 50th anniversary of the invention of our iconic Bubble Wrap� brand cushioning.
This year�s highlighted achievements exemplify the benefits of a balanced approach:
� Performed steadily in our food and medical businesses, which helped to partially offset volume declines in our industrial businesses due primarily to the difficult macroeconomic environment;
� Innovated and launched over 25 new products and won several industry awards worldwide;
� Streamlined our operations, reduced our cost structure and discretionary spending, and successfully operated
our global network on a new, �one instance� SAP enterprise system;
� Achieved record safety, quality and customer service performance levels;
� Opened 2 new facilities, which concludes the construction phase of our Global Manufacturing Strategy,
giving us a manufacturing network well-positioned for long-term growth in developing regions;
� Increased our profitability with a 320 basis point year-over-year improvement in our gross margin and a
340 basis point year-over-year improvement in our operating margin;
� Reported record free cash flow of $501 million1;
� Continued to return value to stockholders through dividends and a reduction in net debt1; and
� Recognized for best-in-class governance practices by GovernanceMetrics International�.
Expanding Profitability
We generated $4.2 billion in sales, and while we experienced a 12% decline versus the prior year (which includes 5% from foreign exchange), we saw sequential sales growth in the second half of the year across most of our businesses. The sales decline in 2009 primarily reflected an 8% decline in volumes, which was largely attributable to our industrial businesses who were most impacted by economic weakness. Despite the volume declines, we held our gross profit relatively steady, while increasing our gross margin by 320 basis points to 28.7%. We did achieve a 24% increase in operating profit, as well as a 340 basis point increase in our operat- ing margin to 11.6% (or 12.0% on an adjusted basis)1�bringing us closer to our operating margin goal of
1) Please refer to the definition and reconciliation pages located before the presentation of the Form 10-K for reconciliations of non-U.S. GAAP financial measures.
1

2
15% by the 2012 to 2013 timeframe. These increases were driven by stabilized input costs and our successful management of price/mix, combined with $80 million of benefits from our global manufacturing strategy and cost reduction and productivity programs.
Making a Global Impact
Over the course of the year, we saw an accelerated rate of volume demand in our developing regions�with �BRIC� country sales improving 18% in the fourth quarter versus the prior year period, which includes 6% of foreign exchange. This growth expanded our developing region sales to 17% of our total net sales in that quarter. This growth continues our long term expansion into international markets where we see the growth of disposable incomes, urbanization and increased demand for higher quality, packaged foods as growth drivers for each of our businesses. Contributing to our growth in developing regions, were the three new green-field sites that we launched under our Global Manufacturing Strategy (GMS). This included the launch of our third and final site in Poland, in early 2009. This launch marks the completion of the construction phase of GMS, as well as bringing in-house the production of our Ethafoam� polyethylene foams for our specialty materials business.
The benefits associated with these new sites, as well as the limited closures and consolidations among our existing platform, brought the program�s full benefit run-rate to $45 million in 2009, which we expect to increase to $55 million in 2010.
Additionally, line expansions and technology upgrades in facilities in Brazil, Russia and Hungary continue to position us for profitable growth in these economies as we expect the recovery in developing regions to outpace advanced regions in the years ahead.
Making a Difference with Innovation
Our commitment to innovation did not waver in 2009. We launched over 25 new solutions, received ongoing industry recognition of our developments, and enjoyed a successful first full year in our new Americas Packforum� facility based in Atlanta, Georgia hosting over 200 customer visits and events.
We remained focused on sustainability as a component of packaging solution development. We introduced our Ethafoam� HRC foam, which uses a minimum of 65% recycled content, as well as our new Cryovac� CT-301TM shrink film, which leverages a new proprietary and patent-pending manufacturing process that allows us to deliver film up to 50% thinner than comparable solutions, while maintaining and even exceeding performance metrics. Of course, these are only 2 of many packaging solutions that we launched this year� with many others highlighted in our 2009 Annual Review.
Making It Happen through Supply Chain Excellence
Our best-in-class supply chain organization continued to advance our operational and financial performance with the use of continuous improvement methodologies and by leveraging our one instance of SAP, which now covers 90% of our revenue worldwide.
These efforts and investments resulted in a number of notable achievements:
� Improved our quality performance by 7%;
� Reduced our total recordable incident rate to 1.04 versus our goal of 1.40;
� Reduced our inventory levels by $95 million;
� Generated measurable benefits from our �cost to serve� analysis in our European food businesses by leverag-
ing our one instance of SAP to reduce our costs and process complexities. This program is now being rolled out to other regions as part of our ongoing dedication to continuous improvement;

� Developed a proprietary, patent-pending new manufacturing technique that enables significant source reduction�used initially in our CT-301TM shrink film; and
� Successfully launched 2 new facilities, expanded line capacity and upgraded technology in several others, while seamlessly consolidating or closing certain operations without service disruption.
Generating Cash and Liquidity
We generated a record $501 million in free cash flow in the year. This achievement was largely due to a $120 million improvement in working capital items and a $100 million reduction in capital expenditures following the completion of the construction phase of our GMS program.
Additionally, we finished the year with approximately $700 million in cash and cash equivalents and approximately $700 million in committed credit facilities. We also reduced our net debt balance by nearly $400 million during the year. We achieved this by generating solid cash flows from operations, raising $700 million in new notes in the first half of the year, redeeming our 3% convertible senior notes, and retiring the balance of our 6.95% senior notes. With approximately $1.4 billion of cash and committed liquidity, we are comfortably positioned for our pending payment under the Settlement agreement relating to the Cryovac transaction and have no significant debt maturities until 2013. Combined with ongoing cash flows from operations and an expected significant cash tax benefit following the Settlement payment, we remain confident about our liquidity position and our ability to continue to reduce debt levels.
Making a Better Tomorrow
As we look to the future, we will continue to capitalize on our heritage, culture and investments in innovation, in our extensive global platform, in our commitment to sustainability and in our global team of industry-leading professionals�who remain unsurpassed in service and measurable value-creation for customers worldwide.
2010 is the next step in this journey and our priorities are:
� Cash flow and improving return on assets;
� Continuing to innovate;
� Optimizing processes and operations to maximize profitability;
� Accelerating our growth in India and China;
� Developing our people�our future leaders; and
� Continuing our sustainability practices�in everything we do to help make the world a better place today
and tomorrow.
Our near term achievements combined with our long term growth strategies position us well to continue
to be a leader in the markets we create and serve. By achieving our goals, we expect to continue to generate measurable value for our customers and stakeholders worldwide. Further demonstrating�it�s not just what we make, it�s what we make happen.
It is for these reasons that all of us at Sealed Air remain passionate about our business, our customers, our communities, and our stockholders.
Sincerely,
William V. Hickey
President and Chief Executive Officer
 3
""",
"""
The past year must be viewed as a significant milestone in the history of M&T Bank Corporation, thanks to the closing, at the start of the second quarter, of the largest merger in our history. That merger, with the Baltimore-
based Allfirst Financial Inc. (Allfirst) expanded our geographic reach, increased by more than half (269) the size of our network of branches, and led to our being ranked among the twenty largest publicly-traded bank holding companies in the United States, as measured by total assets. Indeed, as a result of the merger, total assets grew by 49%, total deposit balances increased by 50% and noninterest income rose by 54%. A full report on the progress of our combination with the former Allfirst follows below. It must be noted at the outset, though, that growth in M&T�s net income and earnings per share in 2003 was driven, in significant part, by asset and deposit growth resulting from the merger, coupled with increased lending to consumers for homes and automobiles. Such factors were tempered by increased expenses for daily operations that reflect our expanded scale, amortization of intangible assets, and costs associated with combining the operations of the former Allfirst with those of M&T.
Using generally accepted accounting principles (GAAP), M&T�s diluted earnings per share in 2003 were $4.95 and net income for last year was $574 million. Those measures improved by 4% and 26% from $4.78 and $457 million, respectively, in 2002. The differential in the rate of increase between the two measures reflects the issuance of 26.7 million common shares on April 1 of last year to Allied Irish Banks, p.l.c. (AIB) in connection with the merger with Allfirst. In addition to the shares, we paid $886 million to AIB to complete the transaction. In return we received $16 billion of assets, including $10 billion of loans and leases, and assumed $14 billion of liabilities, including $11 billion of deposits.
Net income in 2003 represented a rate of return on average assets of 1.27% and on average common stockholders� equity of 11.62%. The comparable rates of return in 2002 were 1.43% and 15.09%.
Since 1998, we have provided supplemental reporting of earnings on a �net operating� or �tangible� basis. In contrast to GAAP-basis results, net operating results exclude the after-tax effect of core deposit and other intangible assets � both in the income statement and on the balance sheet � and merger- related expenses associated with the integration of acquired operations with and into M&T. We have consistently reported in this way to help investors understand the effect of merger activity in M&T�s reported results.
Merger-related expenses last year amounted to $39 million after applicable income tax effect, or 34 cents per diluted share. There were no merger-related expenses in 2002. Non-cash charges for amortization of core deposit and other
5

6
intangible assets, also after income tax effect, totaled $48 million in 2003, or 41 cents per diluted share. In 2002, amortization charges, net of tax effect, were $32 million or 34 cents per diluted share. The increased level of amortization was a direct consequence of the Allfirst merger.
Diluted net operating earnings per share, which exclude merger-related expenses and charges for core deposit and other intangible assets, grew 11% last year to $5.70. In 2002 net operating earnings per diluted share were $5.12. For all shares in the aggregate, net operating income in 2003 totaled $661 million. That was a 35% jump from $489 million a year earlier. The net operating return on average tangible assets slipped a bit, though, as we integrated the acquired operations with those we had previously. That return was 1.55% last year, compared with 1.59% in 2002. The net operating return on average tangible equity rose, however, to 28.49% in 2003 from 26.71% in 2002.
As required by the Securities and Exchange Commission, reconciliations of GAAP-basis net income and net operating income, average total assets and average tangible assets, and average total stockholders� equity and average tangible stockholders� equity appear on page 24 of this report.
Last year also witnessed a significant change in M&T accounting policy. As explained in these pages a year ago, M&T began reporting stock-based compensation as a component of operating expenses in 2003. In implementing the change, M&T not only charged stock-based compensation expense to last year�s results, but elected to restate previously reported financial results to also include stock-based compensation in determining net income for earlier years, as allowed by the accounting rule makers at the Financial Accounting Standards Board. Oftentimes restatements of previously reported financial information are not looked upon favorably. However, in this instance, the restatement clearly enhances the comparability and transparency of the information reported. As a result of our chosen method of implementation, stock-based compensation is recognized as an expense in all GAAP-basis and net operating-basis results presented in this Annual Report. After applicable tax effect, stock-based compensation amounted to $32 million, or 27 cents per diluted share, in 2003 and $28 million, or 29 cents per diluted share, in 2002.
Comparisons of financial statement line item details between years are complicated by the merger of Allfirst into M&T. For example, the pro-rated nine-month impact of the $10 billion added to the company�s loan portfolio on the merger date was the major element in the loan total�s 33% growth from its 2002 average level of $25.5 billion to last year�s $34.0 billion. That $8.5 billion increment also includes a $1 billion rise in the average outstanding balance of

loans to consumers for financing the purchase of automobiles and other vehicles, plus growth of more modest proportions in other loan categories.
The rate profile of loans acquired in the merger contributed greatly to the downward drift in the average taxable-equivalent rate earned on the loan portfolio in 2003. That rate declined last year to 5.61% from 6.57% in 2002. Likewise, the average taxable-equivalent rate earned on the company�s $39.5 billion average total of earning assets declined by nearly the same number of basis points (hundredths of one percent) as the fall in the average rate on loans, moving from 6.42% in 2002 to 5.42% in 2003.
The average rate paid on interest-bearing liabilities went down by 78 basis points last year, from 2.39% to 1.61%. The differing magnitude of decline resulted in a 22 basis point narrowing of the spread between the average rates earned and paid. The net interest spread was 3.81% in 2003 and 4.03% in 2002. The lowered net interest spread curbed year-to-year growth in taxable-equivalent net interest income, the main component of net income. At $1.62 billion last year, it was 28% higher than 2002�s $1.26 billion.
Net charge-offs of loans � that is, the amount by which charge-offs of loans exceeded recoveries of loans previously charged off � declined last year both in dollar total and as a percentage of average loans outstanding. Net charge-offs were $97 million in 2003, or .28% of average loans outstanding. A year earlier, net charge-offs were $108 million, or .42% of the total loan average. There were no significant net charge-offs of loans acquired in the merger with Allfirst during the nine months of 2003 that we operated that franchise.
The ratio of nonperforming loans to total loans also improved last year. At December 31, 2003, nonperforming loans totaled $240 million. That was equal to .67% of the $35.8 billion of loans outstanding. At December 31, 2002, nonperforming loans were $215 million, or .84% of the $25.7 billion of loans then outstanding.
The provision for credit losses in 2003 was $131 million, or $34 million more than net charge-offs. The provision in 2002 was $122 million, some
$14 million more than that year�s net charge-offs. At 2003�s end, the allowance for credit losses � including an existing allowance of $146 million that came with Allfirst�s loans in the merger � stood at $614 million and was equal to 1.72% of outstanding loans. At December 31, 2002, the allowance of $436 million was equal to 1.70% of the then outstanding loan total.
Noninterest income grew significantly � expanding 62% to $831 million last year from $512 million in 2002. Nearly nine-tenths of that increase can be sourced to operations or market areas associated with the former Allfirst franchise.
7

8
The remainder was largely due to higher fee revenues earned from providing deposit account services to customers and from our residential mortgage loan origination and sales activities.
Noninterest operating expenses, which exclude the merger-related expenses and amortization charges, bulged to $1.31 billion last year, 44% above the 2002 total of $910 million. The elevated expenses reflect the breadth of our post- merger operations. A reconciliation of noninterest operating expenses and total noninterest expenses appears on page 24.
Because the operational integration of Allfirst with M&T occurred over the last nine months of 2003, some redundant operations co-existed during that period. As a result, M&T�s efficiency ratio � that is, noninterest operating expenses as described above divided by the sum of taxable-equivalent net interest income and noninterest income (exclusive of securities transactions) � came in at 53.6% in 2003, compared with 51.3% for the prior year.
THE ALLFIRST MERGER: A PROGRESS REPORT
One might say that the merger with Allfirst was an endeavor for which the employees of M&T Bank have been preparing for more than a decade, as we have steadily grown and built our capacity to integrate institutions of larger and larger size with our operations. Indeed, at the time we acquired ONBANCorp in 1998 it was roughly one-third the size of M&T; at the time we acquired Keystone in 2000, it too was about one-third the size of a larger M&T; and, at the time we acquired Allfirst this past year, it was nearly half the size of a still-larger M&T. Clearly the last six years have been a period of ambitious growth for this company; each of the mergers we have completed was larger, in absolute terms, than the last, just as each one was large in relationship to our own size at the time. The Allfirst merger is, in other words, the latest of a series of challenges of steadily increasing magnitude. As we have undertaken each, we have focused on our core values and strengths: our role as a community banker, hiring and retaining top-quality employees, smooth execution of our conversion plan, and attention to value for shareholders. Underlying all, is our commitment to honesty, integrity and transparency.
Still, as well-prepared as we were for it and as much as it resembled, in scale, previous mergers, there can be little doubt that the Allfirst integration has represented both the greatest challenge, as well as the greatest opportunity, in M&T�s history. This is to report both that we have made significant progress toward our goal of profitably incorporating the Allfirst franchise, as judged by such measures as the retention of customers and the reduction in costs � and, too, that much remains to be accomplished.

In discussing Allfirst, it remains important to keep in mind that this merger is, in significant ways, different from those previous, in both quantitative and qualitative regards. Our new markets, for instance, and their population of 8.2 million, not only represent the largest single addition ever to our potential customer base (a 72% increase when compared with the 11.4 million population of our �vintage� markets in upstate New York and central Pennsylvania) but are both growing faster and are more affluent. Their 1.2% annual rate of population growth is four-times the .3% average of our pre-Allfirst market region. Indeed, the 1.1% annual population growth (1990-2002) in Maryland alone compares to just a 0.1% annual increase in upstate New York during the same period. Moreover, despite their relatively smaller current population, our new markets� aggregate net income ($308.4 billion in 2001) virtually matched that of our combined upstate New York and central Pennsylvania markets ($311.2 billion). Put another way, per capita personal income in the �new� markets ($38,060) was 39% higher than that of our vintage markets ($27,450). The promise of these markets is reflected not only in the fact that the Baltimore-Washington area is the nation�s fourth-largest metropolitan statistical area but that five of
the 20 wealthiest counties in the country are located there.
Greater affluence is, of course, good news for any financial institution seeking to attract deposits and offer lending products and other financial services. At the same time, however, a more affluent market is, inevitably, a more competitive one. In entering the Maryland, District of Columbia and northern Virginia markets, we entered a region in which a considerable number of major national and regional competitor banks were already well-established � and were accustomed to spending heavily on advertising to gain and maintain public awareness of their brand names, products and services. In fact, annual spending on bank industry advertising in the Baltimore/Washington markets totaled some $32 million, or three-times the cost of such spending in the Buffalo, Rochester and Syracuse markets combined (19% more on a per capita basis). Needless to say, there was, in the Baltimore-Washington area, little or no awareness at all of the M&T name at the time of our merger with Allfirst � whose own brand name was relatively new to a region which had been more familiar with some of its predecessor institutions, such as the First National Bank of Maryland.
Nor was a need to establish awareness of the M&T name our only challenge. At the time of the merger, the Allfirst cost structure � as measured by its efficiency ratio (70.5%) � compared unfavorably with that of the previously- existing M&T operations (49.8%).
Notwithstanding these significant challenges, we have, to date, much
9

10
more than held our own in the former Allfirst markets. The credit for this must go first, and foremost, to a capable and dedicated workforce inherited from Allfirst itself. More than 4,000 former Allfirst employees � notably those with direct contact with customers � were kept on after the merger. The retained former Allfirst employees � whether in consumer, business or commercial banking, whether in branches or the Baltimore main office � made certain their customers were well-taken care of through the transition period. They have been complemented by hundreds of dedicated M&T employees who joined with them, working long hours far from home or, indeed, permanently relocating to Baltimore and other locations, to help smooth the integration, whether through painstaking conversion of vital information technology systems or by �buddying� with individual Allfirst employees to familiarize them with M&T procedures and culture.
In addition, however, as befits the different nature of the former Allfirst markets, we have taken some new and distinct approaches in our effort to make this merger successful. As a conservatively-run firm, for instance, we had traditionally focused, in our advertising, on the products and services we
make available to potential customers. We recognized, however, that gaining recognition for the M&T name such that our message would be heard in our new markets was a notable challenge. A heavy investment in conventional advertising might have borne fruit only slowly � and ran the risk of not sufficiently distinguishing ourselves from our well-established competitors.
It was in this context that we entered into an agreement with Baltimore�s National Football League team, the Ravens, to purchase the naming rights to what had been known simply as Ravens Stadium. It was our hope that what initially appeared to us to be a high price � $75 million over 15 years � would prove to be a bargain, when compared with the cost of other avenues to establish the M&T name. There is good reason to believe that this has, indeed, proven to be the case. In September 2003 � just four months after the naming rights announcement, and just five months after the closing of the Allfirst transaction � M&T found that our �unaided top-of-mind brand awareness� in the Baltimore region went from virtually nil to third place among 18 financial institutions included in our survey. Indeed, over the course of the past professional football season, M&T Bank Stadium became familiar to television viewers and sports page readers across the country, through broadcasts and reports of Ravens� home games. Professional football, it�s worth noting, is ranked as the number one spectator sport in the U.S., with some 100 million fans.
Although the stadium naming rights were, from a marketing point of view, a departure from our past practice, they should, in a larger sense, be

understood as much in keeping with the traditional (and traditionalist) M&T approach to our business. We have always considered and presented ourselves as a community bank � a �hometown� bank, if you will � one interested in the needs and issues of the places we serve. So it has been in Baltimore and other parts of the former Allfirst markets, in ways small and large. The M&T Bank All Community Team, for instance, brings together the Ravens and Baltimore�s business leadership to support fundraising and volunteer projects to benefit Baltimore. Our long-standing interest in supporting education is reflected in the advent of the M&T Bank Honor Rows � through which 150 children from socially and economically disadvantaged backgrounds receive free football tickets, as a reward for academic performance. The M&T-donated �Believemobile� has allowed the city of Baltimore to bring entertainment on a professional soundstage to some of the city�s poorest neighborhoods. Working with a local non-profit, Operation Hope, 30 of our employees have made a four-week volunteer commitment to teach financial literacy in the Baltimore public schools. And we have contributed $1 million towards the renovation of Baltimore�s historic, newly-reopened Hippodrome Theatre. Not only will this historic renovation provide the city with a world-class venue for Broadway-quality theatre but will, we believe, help maintain the momentum of renewal ongoing on Baltimore�s west side. Thus we hope to have played a role not only in improving the city�s quality of life but in sparking economic development.
Broadly, it was our belief that our competitors were viewed as national banks; we set out to be in our new markets what we have always been in our vintage markets � a local bank. To do so, our management team (led by Eugene Sheehy and Atwood Collins, III � the latter a long-time member of the M&T management team who also has deep family roots in Baltimore) has worked diligently to link the bank with prominent and effective local organizations � ranging from an effort to improve minority entrepreneurship to the campaign to preserve Maryland�s greatest natural resource, the Chesapeake Bay.
But no forms of public relations and community service would bear fruit were we not able to deliver consistently good service to our new customers. Mergers are demanding, often in unexpected ways. We�ve had to work to
be able to give customers what they want, even if that means making quick adjustments � such as tripling our e-mail response staffing so as to successfully handle inquires from customers in the first days after the Allfirst conversion, or quickly fixing the problem that was making it difficult for customers to activate M&T Business Visa Cards. Our commercial banking business, in particular, faced special challenges. Allfirst had emphasized fee revenue by providing a broad
11

12
array of customized cash management and depository services to large commercial and government customers. On these dimensions, Allfirst�s commercial banking business was significantly larger than M&T�s. In order to retain such customers more than 120 separate Allfirst commercial products had to be made consistent with M&T practices. Some 99,000 person-hours were required just for information technology adaptations. It was an investment we did not hesitate to make.
We well appreciate that our responsiveness, or lack of it, as well as maintaining consistent service even in such relatively small matters throughout the course of a merger conversion, does much to establish our overall long-term reputation in our new markets.
More broadly, the merger of M&T and Allfirst has been, in a number of ways, a merger of businesses with their own comparative advantages. The international trade expertise of Allfirst has now been complemented by M&T�s expertise in residential mortgage finance, for instance; to M&T�s familiarity with automobile floor plan financing has been added Allfirst�s knowledge of lending to government contractors. It�s our hope and belief that, through such complements, the whole of this larger company will be greater than the sum of its previous parts.
The early results of this, our largest merger yet, are, indeed, promising. Six months after the conversion of customer data to M&T�s operating systems, consumer demand deposit and NOW account balances in the former Allfirst markets had grown by 7.6%, to $2.1 billion. In those same markets, checking account sales in 2003 exceeded those of 2002 by 12.7%. What�s more, M&T has successfully retained former Allfirst households. In the first five months following conversion � arguably a period of vulnerability for a new entrant to the market � some 96.5% of consumer households remained M&T customers (compared with 95.8% in our vintage markets over the same period). Significantly, during that same time frame we have retained 98.1% of the most profitable former Allfirst households, compared to 96.0% in our vintage markets. We were similarly successful in retaining commercial business. Although we chose not to continue to originate loans in several specialized areas not in keeping with our traditional focus (such as maritime lending), nonetheless, virtually all Allfirst�s large corporate, middle market, commercial real estate and government banking customers continue to bank with M&T. By all these counts, we have improved upon our performance during the initial six-month period following our last major merger, that of Keystone Financial in October 2000. Notwithstanding their impressiveness, such figures, nonetheless mask the enormity of the task and the accomplishment; Allfirst, after all, brought us some 1.4 million customers who needed to be served and served well.

In addition to maintaining and expanding our customer base, we have made progress in controlling costs in the former Allfirst franchise. We estimate that, as of December 31, 2003, we had realized the run-rate cost savings which we estimated at the time of the merger announcement. This is not, however, to say, that this work is done.
In short, our experience, to date, in the markets we have entered as a result of the Allfirst merger, has done nothing to shake and everything to confirm our initial confidence in the most ambitious expansion effort in M&T history. Nevertheless, we do not underestimate the extent of the continuing effort which long-term success will require.
CORPORATE GOVERNANCE
Over the past year M&T has not only begun to do business in important new markets but to do so under significant new rules. Landmark federal legislation, the 2002 Sarbanes-Oxley Act, passed �to improve the accuracy and reliability of corporate disclosures� is, by many measures, the most sweeping new regulation in corporate governance since the Securities Exchange Act of 1934. Its emphasis on accurate accounting information through requirements for independent membership majorities on boards of directors and the personal liability of chief executive and chief financial officers for any financial statement misrepresentation were not unexpected in the light of the corporate accounting scandals which have made brand names like Enron, Adelphia and WorldCom synonymous with scandal.
It should go without saying that M&T has done its utmost to come into full compliance with the new law � and that doing so does not represent any change in our historic corporate culture. A majority of the members of our board and its audit committee have historically been outside directors. Just as we have hewed to traditional banking practices, so have we adhered to conservative accounting and transparent reporting methods � reporting earnings, as noted above, on both a GAAP and non-GAAP basis so as to make year-over-year comparisons as clear and easy as possible. Investors, moreover, will recall that we became (in 2002) one of the first banks of our size to announce that we would account for stock options as expenses in our income statement. Shareholders should be assured that this change is no mere window-dressing. Where once we considered stock options primarily as a compensation-related number � that is, how many were to be issued? � we now consider them more so in value terms and focus our attention on the need to cover, with revenue, the costs which they represent.
We have made such changes because we well understand � and deeply
13

14
believe � that it is only in a business world in which profits and losses are clearly stated, and in which the interests of stockholders are paramount for management, that capital markets � of which banking is so central a part � can properly function. Accounting transparency and independent oversight are the oxygen
of free enterprise.
All this said, however, I cannot help but to draw attention to the reality
that, through this new generation of regulation, corporate wrongdoing has had the effect of imposing significant costs and burdens both on the overwhelming majority of firms which have always been open and honest, and, indirectly, on those firms� customers. At M&T, we estimate that the overall cost of complying with all federal and state regulations exceeded $50 million in 2003. That represented about four percent of the company�s operating expenses. More than 500 full-time equivalent employees contributed to our compliance efforts. Sarbanes-Oxley has added yet another layer of costs to this total. Specifically, the new legislation caused our compendium of corporate governance standards to grow from a listing on two pages to a published program fully a half-inch thick. It catalogues our means of complying with no less than 294 duties and obligations.
Most of the new legislation�s requirements are both unobjectionable
and in keeping with common sense. Some, however, are not. For instance, the rule that anyone employed by the company or its affiliates, in any capacity, is prohibited from service on its audit committee effectively precludes the chief financial officer of AIB from such service. Yet I can think of no one with any greater interest in ensuring that this company is both honest and profitable, lest AIB put its own substantial investment at risk. An independent majority on the audit committee makes sense; excluding a financial expert simply because he is employed by a major shareholder may not actually provide his fellow shareholders with the best protection. This is not to say that key members of a firm�s management should sit on its audit committee, but it does seem foolish to exclude from M&T�s audit committee someone, in the person of AIB chief financial officer Gary Kennedy, who is arguably a highly-qualified financial expert with particular knowledge of banking.
We must take great care lest demands for both independence and commitment on the part of board members � particularly those on the audit committee, whose duties are highly-demanding � drive qualified candidates away from such service altogether. Already, those on M&T�s own audit committee find their role constitutes something close to a part-time job all of its own, in addition to executive roles they play in their own firms. It is a role with significant pressures: they find they must now be concerned about their

own personal liability, for instance, when giving their approbation to financial representations. New demands risk diluting the quality of those willing and able to serve.
Nor are such contraventions of common sense limited to legislation or regulatory fiat. Consider, for instance, new New York Stock Exchange regulations which call for a firm�s directors to engage in an annual �self-evaluation� process, wherein they are to assess the performance of the board and its committees. Such an ill-advised requirement for what amounts to corporate group therapy not only seems silly but may well be counter-productive and risks inhibiting candor on the part of directors. It is, in other words, a rather Stalinesque recipe for group-think, rather than a tonic to induce independent expression.
Concern about corporate governance has also spawned private independent rating agencies which set their own standards and issue their own report cards for individual firms, as a service to investors. This is generally to the good.
But in some instances, such agencies adopt an unfortunate, one-size-fits-all approach which ill-serves those in whose interest they nominally operate.
M&T, for instance, has had to struggle to convince outside observers that
the size of our board of directors � 26 � is not designed somehow to dilute
its influence and to advance the interests of management at the expense of investors (an oxymoron in our case, given the shareholdings of our managers). In fact, the size of our board is nothing less than a prudential means to
protect the interests of shareholders. As a result of our long series of mergers, we have added as directors, one-time directors of ONBANCorp, Keystone, AIB and other institutions now joined into M&T. We believe that their industry experience and knowledge of their local markets is invaluable. That this makes for a board of more than 16 members (the measure some have set) should lead to concern only if one believes that all firms should be judged by the same standards, regardless of their special circumstance. No careful analyst would adopt such an approach � and neither should corporate governance rating agencies or pension funds.
These are not, in themselves, matters of great significance � and, indeed, people of good faith may disagree with my specific conclusions. I raise them, rather, to make the larger point that those setting corporate governance standards and requirements must seek to strike a balance among interests. It is crucially important for lawmakers, for instance, to keep in mind, as they go forward, that added regulation increases costs and that those corporate wrongdoers now under indictment or otherwise disgraced are in that position because they violated laws that were already on the books � that is, before Sarbanes-Oxley was approved.
15

16
Indeed, of approximately 10,500 publicly-held companies in the United States, only 225 had to restate their financial reports in 2001, the last full year prior to the passage of Sarbanes-Oxley. Moreover, from 1997 through 2002, the Securities and Exchange Commission found it necessary to file 515 enforcement actions against 164 corporate entities � representing less than 2% of all publicly- held companies in the U.S. It is true, and terribly unfortunate, that this number has been increasing. But it is also true that this relatively small group of wrongdoers has set in motion a wave of more stringent regulation which has imposed significant costs on the overwhelming majority of firms which have played by the rules. Nor can we be assured that the new, stricter rules will
yield any better results than the previous regime. It is not as if there were not independent board members and outside auditors serving firms which are now in disgrace. Laws, after all, are the necessary but not sufficient means to deter those who, for reasons of greed or desperation, choose to commit fraud or to loot corporate treasuries for their personal gain. But, at the same time, rules cannot substitute for honor and ethics, and the understanding that, without these, the functioning of markets and the wealth and well-being of all are threatened.
Still, even though the extent of corporate wrongdoing has been
relatively limited, it has clearly grown alarmingly and the dimensions of specific misconduct have been outrageous, with innocent shareholders and employees victimized. In this context, it is likely that we will continue in a period of more regulation, not less. This is to express the hope that legislators keep in mind that more regulation will also mean higher costs � without any certainty of serving the public interest.
NOTE ON BUFFALO
In years past, I have had a good deal to say in these messages about public affairs in M&T�s headquarters region, that of Buffalo and Western New York, especially as regards high levels of taxation, ineffective and expensive delivery of public services, and loss of population and economic vitality. The civic, community and economic health of Buffalo and environs continue to be of great personal importance to me. In that light, I was privileged, this past year, to be named, by New York Governor George Pataki, to serve on the Buffalo Fiscal Stability Authority, the so-called �control board� which is reviewing the city�s expenditures and revenues in an effort to set it on a sustainable, long-term financial course. That official position makes it inappropriate, in my view, to expound such topics as the city�s finances, as in previous messages. Moreover, as a result of our mergers of the past several years, Buffalo � although of ongoing personal importance to

 me � is no longer as central to the fortunes of M&T. Buffalonians can rest assured that M&T will continue to call the city home, for our corporate headquarters. But readers of these pages can now be spared the jeremiads of years past.
The past year saw the passing of three former members of our boards of directors. Barber Conable, former member of Congress and president of the World
Bank (1986-1991), was a former director of M&T Bank Corporation and M&T Bank, as well as Chairman of M&T Bank�s Directors Advisory Council � Rochester Division. We will remember his distinguished service to his country and to this company with gratitude.
William H. Harder was a former member of the board of M&T Bank. He had served as president of Buffalo Savings Bank. We will remember his dedication to the welfare of his native Buffalo and his service to the city on numerous charitable boards and fund drives. Buffalo will miss him greatly.
Raymond D. Stevens, Jr., was a former director of M&T Bank Corporation and a member of its executive committee, and a director of M&T Bank, and a member of both its executive, and trust and investment committees. He was the retired chairman of the board of Pratt & Lambert United, Inc. We recall his sound advice with appreciation. Mr. Stevens served as a director from 1963 until his retirement in 1998, the longest such tenure of service in M&T�s history.
Finally, this message would not be complete without noting the retirement, this past year, of M&T Executive Vice President for Human Resources Ray Logan. In his 17 years in that post, Ray helped shepherd a workforce which grew from 2,300 when he began to 14,000 at the time of his retirement. He played a key role in our recruitment and retention of top-quality employees and worked hard, and creatively, to ensure that, as the company grew, employees continued to feel that their voices were heard and their ideas taken seriously. It is a legacy
Chairman of the Board,
President and Chief Executive Officer
of which we will strive to be mindful.
Robert G. Wilmers
 February 18, 2004
17
"""



]


np_test_sentences = np.array(test_sentences)
test_sequences = tokenizer.texts_to_sequences(np_test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')



predictions = model.predict(test_padded).tolist()

for prediction in predictions:
  prediction_value = prediction[0]
  int_prediction = int(round(prediction_value*5))
  print(int_prediction)

